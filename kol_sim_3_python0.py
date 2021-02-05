"""
方法2.2.1 python version

策略：只对新公众号进行添加及其余公众号更新。
使用 基于tfidf方法，使用Inverted Index 减少内存消耗
"""
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import pymysql.cursors
from datetime import datetime, date, timedelta
import logging
import collections
import math


logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='kol_sim_3_python0.log',
                    filemode='w')

logger = logging.getLogger(__name__)


# 读取停用词文件
def read_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        txt = f.read()
    return txt


# 中文预处理（返回值：分词后的df，以及出现的自定义关键词）
def processing(param_df):
    word = list(jieba.cut(param_df['article']))
    stop_word = set(read_stopwords('./hit_stopwords.txt'))
    res, is_keyword = [], set()
    for w in word:
        if not w or len(w) == 0 or len(w) == 1 or w in stop_word:
            continue
        if w.isnumeric():
            continue

        w = w.lower()
        res.append(w)

    return " ".join(res), ",".join(list(is_keyword))


def sql_init():
    connection = pymysql.connect(host='192.168.100.54',
                                 user='test',
                                 password='Meihao100@bfbd',
                                 database='kol',
                                 charset='utf8mb4'
                                 )
    return connection


def read_db(start_time):
    df = pd.DataFrame(columns=('official_account', 'longorshort', 'article'))
    dict_stat = dict()

    connection = sql_init()
    with connection.cursor() as cursor:
        # sql_select = "select c.content, c.create_time , a.official_account_name from kol_content as c inner join kol_article as a on c.third_id = a.third_id where a.type_id = 1 and c.create_time >= "+str(start_time)
        sql_select = "select c.content, c.create_time , a.official_account_name from kol_content as c inner join kol_article as a on c.third_id = a.third_id where a.type_id = 1 LIMIT 70"
        cursor.execute(sql_select)
        data = cursor.fetchall()
        logger.info("start to store content...")
        print("start to store content...")
        for i in data:
            content, time, author = str(i[0]), i[1], i[2]
            islong = True if len(content) > 500 else False
            df = df.append({'official_account': author, 'longorshort': islong, 'article_index': -1, 'article': content}, ignore_index=True)

    connection.ping(True)
    with connection.cursor() as cursor:
        logger.info("start to store statistic...")
        print("start to store statistic...")
        sql_select_stat = "select koa.name, koa.organization_id, koa.all_article_average_read_statistics, koa.all_article_average_like_statistics, koa.grab_article_statistics, koa.newest_article_publish_time, koa.create_time, koa.update_time from kol_official_account as koa"
        cursor.execute(sql_select_stat)
        data = cursor.fetchall()
        for j in data:
            dict_stat[str(j[0])] = j[1:]

    return df, dict_stat


# 加载以及更新倒排索引字典，字典格式：{word1: [[name1, name2, ...], [tf1, tf2, ...]], ...} 单词->bao han ci dan c
def calculate_inverted_index_vocab(df):
    print('start to update inverted index vocab...')
    vocab = np.load('inverted_index_vocab.npy', allow_pickle=True).item()

    for i in range(len(df)):
        text = (df.loc[i, 'words']).split(' ')
        name = df.loc[i, 'official_account']
        temp_dict = collections.Counter(text)
        for k, v in temp_dict.items():
            if k in vocab:
                arr = vocab[k][0]
                if name in arr:
                    ind = arr.index(name)
                    vocab[k][1][ind] += v/len(text)
                else:
                    vocab[k][0].append(name)
                    vocab[k][1].append(v/len(text))
            else:
                vocab[k] = [[], []]
                vocab[k][0].append(name)
                vocab[k][1].append(v/len(text))

    np.save('inverted_index_vocab.npy', vocab, allow_pickle=True)

    return vocab



def calculate_similarity(stat, vocab, df):
    print('start to calculate similarity...')
    res = pd.DataFrame(columns=('A', 'B', 'similar', 'islong'))
    total = len(stat)
    df = df.groupby('official_account')
    for name, others in df:
        text_list = list(others['words'])
        text = (' '.join(text_list)).split(' ')
        temp_dict = collections.Counter(text)
        sim_account_dict = {} # {name: [分子, 分母]}, cosine = 分子/分母
        current_mode = 0
        # 计算公众号tfidf向量
        # 当前公众号的每个单词，及对应的tf
        for k, tf in temp_dict.items():
            if k in vocab:
                # 该词的tfidf值
                idf = math.log(total/len(vocab[k][0]), 10)
                current_tfidf = (tf/len(text)) * idf
                current_mode += current_tfidf**2

                # 其余公众号对应该词的tfidf
                name_list, tf_list = vocab[k][0], vocab[k][1]
                for j in range(len(name_list)):
                    sim_name = name_list[j]
                    if sim_name == name:
                        continue
                    if sim_name in sim_account_dict:
                        sim_tfidf = tf_list[j] * idf
                        sim_account_dict[sim_name][0] += sim_tfidf * current_tfidf
                    else:
                        sim_account_dict[sim_name] = [0, 0]
                        sim_tfidf = tf_list[j] * idf
                        sim_account_dict[sim_name][0] = sim_tfidf * current_tfidf

                    sim_account_dict[sim_name][1] += sim_tfidf**2

        for sim, value in sim_account_dict.items():
            similarity = value[0]/(math.sqrt(value[1])*math.sqrt(current_mode))
            res = res.append({'A': name, 'B': sim, 'similar': similarity, 'islong': True}, ignore_index=True)

    return res


def update_similarity(res):
    result = res.groupby('B')


def main():
    # 多长时间更新一次kol_similar
    interval = 3
    start_time = date.today() + timedelta(days=-interval)

    # 读取过去n天内的数据
    df, dict_stat = read_db(start_time)

    # 中文预处理
    logger.info("start to pre-processing...")
    print("start to pre-processing...")
    df["temp"] = df.apply(processing, axis=1)
    df[["words", "defined_keywords"]] = df["temp"].apply(pd.Series)

    # 更新 Inverted index vocab
    vocab = calculate_inverted_index_vocab(df)

    # 新数据相似度计算
    result = calculate_similarity(dict_stat, vocab, df)

    # 旧数据相似度更新


    for account, sim_account in result:
        print("########")
        sim_acc, value, islong = list(sim_account['B']), list(sim_account['similar']), list(sim_account['islong'])
        for i in range(len(sim_account)):
            print(account, sim_acc[i], value[i])


if __name__ == "__main__":
    main()