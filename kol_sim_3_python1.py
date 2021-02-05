"""
方法2.2.0：python version

策略：
1. 定期对公众号相似度进行更新

建模方法：(效果最好的是基于tfidf的方法)
1. tfidf
2. keywords embedding
3. doc2vec (word2vec)
4. LDA
5. tfidf + LDA (reinforce probability distribution)

相似度计算方法：
1. MIPS (dense matrix and sparse matrix to avoid out of memory)
2. LSH (效果不好由于数据需要循环加载)

避免1过多：
1. jaccard
"""
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import pymysql.cursors
from datetime import datetime
import logging
from gensim.models import word2vec

from lshash.lshash import LSHash
from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql import functions as F

from gensim.models import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from gensim.matutils import cossim
from gensim import models

# import os
# os.environ['PYSPARK_PYTHON']='/usr/local/bin/python3'

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='kol_sim_3_python.log',
                    filemode='w')

logger = logging.getLogger(__name__)


# 自定义关键词字典 用于Jaccard similarity
defined_keywords_set = {'python', 'java', 'C', 'C++', 'web', 'cv',
                        'NLP', '运营', '算法', '大数据分析', '大数据开发',
                        '数据分析', '产品', '设计', '竞品', '己品', '运维',
                        '职业', '职场', '就业', '教育', '求职', 'hr'}

# 加载训练后的词向量
pretrained_model = word2vec.Word2Vec.load('trained-word2vec-model/pretrained_word2vec.model')


# 数据库连接
def sql_init():
    connection = pymysql.connect(host='192.168.100.54',
                                 user='test',
                                 password='Meihao100@bfbd',
                                 database='kol',
                                 charset='utf8mb4'
                                 )
    return connection

# def sql_init():
#     connection = pymysql.connect(host='192.168.2.157',
#                                  user='kkb_kol',
#                                  password='O7YEj8LDxvuW1g1o',
#                                  port=3306,
#                                  database='kol',
#                                  charset='utf8mb4'
#                                  )
#     return connection

# def sql_init():
#     connection = pymysql.connect(host='192.168.30.200',
#                                  user='root',
#                                  password='qg1xzGZSZt',
#                                  database='kol',
#                                  charset='utf8mb4'
#                                  )
#     return connection


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
        if w in defined_keywords_set:
            is_keyword.add(w)

        res.append(w)

    return " ".join(res), ",".join(list(is_keyword))


# 读数据库（返回值：分词前的df，以及公众号统计信息）
def read_db():
    df = pd.DataFrame(columns=('official_account', 'longorshort', 'article'))
    dict_stat = dict()

    connection = sql_init()
    with connection.cursor() as cursor:
        sql_select = "select c.content, a.official_account_name from kol_content as c inner join kol_article as a on c.third_id = a.third_id where a.type_id = 1"
        cursor.execute(sql_select)
        data = cursor.fetchall()
        logger.info("start to store content...")
        print("start to store content...")
        for i in data:
            content, author = str(i[0]), i[1]
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


def calculate_tfidf(df):
    logger.info("start to calculate tfidf...")
    print("start to calculate tfidf...")
    size = 5000
    CV = CountVectorizer(max_features=size)
    tf = CV.fit_transform(list(df["words"]))
    logger.info('tf calculation successful.')
    TFIDF = TfidfTransformer()
    tfidf = TFIDF.fit_transform(tf)
    logger.info('tfidf calculation successful.')
    vocab = dict(zip([i for i in range(len(CV.get_feature_names()))], CV.get_feature_names()))

    tf_list = []
    # for i in range(tf.shape[0]):
    #     index, value = tf[i].tocoo().col, tf[i].tocoo().data
    #     vector = np.zeros(size)
    #     for k in range(len(index)):
    #         vector[index[k]] = value[k]
    #
    #     tf_list.append([(i, vector[i]) for i in range(size)])

    return tf_list, tfidf, vocab


def calculate_doc2vec(df):
    print("start to calculate doc2vec...")
    conf = (SparkConf().set("spark.cores.max", "16")
            .set("spark.driver.memory", "16g")
            .set("spark.executor.memory", "16g")
            .set("spark.executor.memory_overhead", "16g")
            .set("spark.driver.maxResultsSize", "0"))

    sc = SparkContext(appName="testRDD", conf=conf)
    spark = SparkSession(sc)

    sd = spark.createDataFrame(df)
    sd = sd.withColumn('words_splits', F.split(sd.words, " "))

    word2vec = Word2Vec(vectorSize=50, minCount=0, inputCol="words_splits", outputCol="article_vector")
    model = word2vec.fit(sd)
    df_word2vec = model.transform(sd)
    result = df_word2vec.select('official_account', 'longorshort', 'article_vector').toPandas()
    return result


def train_lda_model(tf, vocab):
    print("start to train LDA model...")
    logger.info("start to train LDA model...")
    lda = LdaModel(corpus=tf, id2word=vocab, num_topics=20)
    lda.save('trained-LDA-model/LDA.model')
    topic_list = lda.print_topics(20)


# 以字典形式存储tfidf sparse tfidf
def calculate_official_account_sparse_tfidf(res, df):
    print("start to calculate official account tfidf vector...")
    logger.info("start to calculate official account tfidf vector...")
    result_article_df = pd.DataFrame(columns=["official_account", "longorshort", "article_vector"])

    for i in range(res.shape[0]):
        account, islong, tfidf = df.loc[i, 'official_account'], df.loc[i, "longorshort"], res[i, :]
        temp_index, temp_data = tfidf.tocoo().col, tfidf.tocoo().data
        vector = dict()
        for k in range(len(temp_index)):
            vector[temp_index[k]] = temp_data[k]
        result_article_df = result_article_df.append({"official_account": account, "longorshort": islong, "article_vector": vector}, ignore_index=True)

    result_official_account_df = pd.DataFrame(columns=["official_account", "longorshort", "embedding"])

    for inf, other in result_article_df.groupby("official_account"):
        islong = True if True in set(other['longorshort']) else False
        vector = dict()
        for temp in other['article_vector']:
            temp_index, temp_data = list(temp.keys()), list(temp.values())
            for k in range(len(temp_index)):
                if temp_index[k] in vector:
                    vector[temp_index[k]] += temp_data[k]
                else:
                    vector[temp_index[k]] = temp_data[k]

        result_official_account_df = result_official_account_df.append({"official_account": inf, "longorshort": islong, "embedding": vector}, ignore_index=True)

    return result_article_df, result_official_account_df


# 以向量形式存储tfidf dense tfidf
def calculate_official_account_dense_tfidf(res, df):
    print("start to calculate official account tfidf vector...")
    logger.info("start to calculate official account tfidf vector...")
    result_article_df = pd.DataFrame(columns=["official_account", "longorshort", "article_vector"])

    for i in range(res.shape[0]):
        account, islong, tfidf = df.loc[i, 'official_account'], df.loc[i, "longorshort"], res[i, :]
        tfidf_index, tfidf_data = tfidf.tocoo().col, tfidf.tocoo().data
        vector = np.zeros(res.shape[1])
        for k in range(len(tfidf_index)): vector[tfidf_index[k]] = tfidf_data[k]

        result_article_df = result_article_df.append(
            {"official_account": account, "longorshort": islong, "article_vector": vector}, ignore_index=True)

    result_official_account_df = pd.DataFrame(columns=["official_account", "longorshort", "embedding"])

    for inf, other in result_article_df.groupby("official_account"):
        islong = True if True in set(other['longorshort']) else False
        result_official_account_df = result_official_account_df.append(
            {"official_account": inf, "longorshort": islong,
             "embedding": np.mean(np.array(other['article_vector']), axis=0)}, ignore_index=True)

    return result_article_df, result_official_account_df


def calculate_official_account_keyword_embedding(res, vocab, df, topk):
    print("start to calculate keyword embedding...")
    result_article_df = pd.DataFrame(columns=["official_account", "longorshort", "article_vector"])

    for i, row in df.iterrows():
        temp = list(zip(res[i].tocoo().col, res[i].tocoo().data))
        temp = sorted(temp, key=lambda x: x[1], reverse=True)

        if len(temp) > topk:
            result, size = temp[:topk], topk
        else:
            result, size = temp, len(temp)

        article_vector = np.zeros(100)
        for j in range(0, size):
            tfidf, keyword, defined_keyword = result[j][1], vocab[result[j][0]], row.defined_keywords
            if keyword in pretrained_model:
                article_vector += np.array(pretrained_model[keyword]*tfidf)
            else:
                continue

        result_article_df = result_article_df.append({"official_account": str(row.official_account), "longorshort": row.longorshort, "article_vector": article_vector/size}, ignore_index=True)

    result_official_account_df = pd.DataFrame(columns=["official_account", "longorshort", "embedding"])

    temp_set = set()
    for inf, vec in result_article_df.groupby(["official_account", "longorshort"]):
        if inf[0] in temp_set:
            continue
        else:
            temp_set.add(inf[0])
            result_official_account_df = result_official_account_df.append({"official_account": str(inf[0]), "longorshort": inf[1], "embedding": np.mean(np.array(vec['article_vector']), axis=0)}, ignore_index=True)

    return result_article_df, result_official_account_df


def calculate_official_account_topic_embedding(tf, df):
    lda = models.ldamodel.LdaModel.load('trained-LDA-model/LDA.model')
    print("start to calculate topic embedding...")
    result_article_df = pd.DataFrame(columns=["official_account", "longorshort", "article_vector"])
    for i in range(len(tf)):
        account, islong, vec = df.loc[i, 'official_account'], df.loc[i, 'longorshort'], lda[tf[i]]
        result_article_df = result_article_df.append({'official_account': account, 'longorshort': islong, 'article_vector': vec}, ignore_index=True)

    result_official_account_df = pd.DataFrame(columns=["official_account", "longorshort", "embedding"])

    for inf, vec in result_article_df.groupby(['official_account', 'longorshort']):
        result_official_account_df = result_official_account_df.append({'official_account': inf[0], 'longorshort': inf[1], 'embedding': list(vec['article_vector'])}, ignore_index=True)

    return result_article_df, result_official_account_df


def calculate_official_account_doc2vec_embedding(df):
    article_vector = calculate_doc2vec(df)
    result_official_account_df = pd.DataFrame(columns=['official_account', 'longorshort', 'embedding'])

    for inf, vec in article_vector.groupby(["official_account", "longorshort"]):
        result_official_account_df = result_official_account_df.append(
            {"official_account": inf[0],
             "longorshort": inf[1],
             "embedding": np.mean(np.array(vec['article_vector']), axis=0)
             }, ignore_index=True)

    return article_vector, result_official_account_df


# LSH and MIPS
def calculate_cosine_lsh(tr_final):
    print("start to calculate similarity based on cosine...")
    logger.info("start to calculate similarity based on cosine...")
    vector_size = tr_final.loc[0, 'embedding'].shape[0]
    result_list = []
    lsh = LSHash(2, vector_size, num_hashtables=1)
    # input
    for i in range(len(tr_final)):
        lsh.index(tr_final.loc[i, 'embedding'], extra_data=tr_final.loc[i, 'official_account'] + ',' + str(tr_final.loc[i, 'longorshort']))
    # output
    for i in range(len(tr_final)):
        temp_set = set()
        account = str(tr_final.loc[i, 'official_account'])

        res = lsh.query(tr_final.loc[i, 'embedding'], num_results=100, distance_func='cosine')

        for j in range(len(res)):
            temp = res[j][0][1].split(',')
            if str(temp[0]) in temp_set:
                continue
            else:
                temp_set.add(str(temp[0]))
                result_list.append({'A': account, 'B': str(temp[0]), 'similar': 1 - res[j][1], 'islong': bool(temp[1])})

    result = pd.DataFrame(result_list).groupby('A')
    return result


# 调用sparse tfidf
def calculate_cosine_sparse_matrix_multiple(tr_final, defined_keywords_dict):
    print("start to calculate similarity based on cosine...")
    logger.info("start to calculate similarity based on cosine...")
    start = datetime.now()
    result_list = list()
    topk = 100
    index = list(defined_keywords_dict.values())
    name_list, islong_list, vector_list = list(tr_final['official_account']), list(tr_final['longorshort']), list(tr_final['embedding'])

    for i in range(len(name_list)):
        acc_key, acc_tfidf = set(vector_list[i].keys()), np.array(list(vector_list[i].values()))
        cosine = []
        for j in range(len(name_list)):
            sim_acc_key, sim_acc_tfidf = set(vector_list[j].keys()), np.array(list(vector_list[j].values()))
            overlap_key = acc_key.intersection(sim_acc_key)
            # key重叠对应的value相乘，然后求和除模。
            similar = sum([vector_list[i][ok]*vector_list[j][ok] for ok in overlap_key]) / (np.linalg.norm(acc_tfidf)*np.linalg.norm(sim_acc_tfidf))
            # 使用Jaccard如果公众号不同但相似度为1。
            if similar == 1 and str(name_list[i]) != str(name_list[j]):
                if len(acc_key.intersection(sim_acc_key)) > 0:
                    temp = 0.3*similar + 0.7*(len(acc_key.intersection(sim_acc_key))/len(acc_key.union(sim_acc_key)))
                    similar = temp if temp != 1 else 0.95
                else:
                    similar = 0.95

            cosine.append(similar)
        temp = list(zip(name_list, islong_list, cosine))
        final = sorted(temp, key=lambda x: x[2], reverse=True)[:topk]

        for value in final:
            result_list.append({'A': str(tr_final.loc[i, 'official_account']), 'B': value[0], 'similar': float(value[2]), 'islong': value[1]})

    result = pd.DataFrame(result_list).groupby('A')
    end = datetime.now()
    print("time-consuming: ", end-start)
    logger.info("calculate similarity successful! time:"+str(end-start))
    return result


# 调用dense tfidf
def calculate_cosine_dense_matrix_multiple(tr_final, defined_keywords_dict, batch):
    print("start to calculate similarity based on cosine...")
    logger.info("start to calculate similarity based on cosine...")
    start = datetime.now()
    result_list = list()
    topk = 100
    index = list(defined_keywords_dict.values())
    name_list, islong_list = list(tr_final['official_account']), list(tr_final['longorshort'])

    # normalize tfidf matrix.
    base = np.array(list(tr_final['embedding']))
    base_temp = np.linalg.norm(base, axis=1).reshape((-1, 1))
    base_norm = base / base_temp

    # 获取索引对应的tfidf并将其转换为01
    base_jaccard = np.where(base[:, index] > 0, 1, 0)

    size = len(tr_final)

    for i in range(size):
        query = np.array(tr_final.loc[i, 'embedding']).reshape((-1, 1))
        query_temp = np.linalg.norm(query)
        query_norm = query / query_temp
        cosine = []

        # 避免内存溢出分批载入base_norm
        if isinstance(batch, int) and batch > 1:
            base_size = base_norm.shape[0]
            batch_size = int(base_size / batch)
            for k in range(0, base_size, batch_size):
                base_norm_temp = base_norm[k:k+batch_size, :]
                cosine += list((base_norm_temp @ query_norm))
        else:
            cosine = list((base_norm @ query_norm))

        temp = list(zip(name_list, islong_list, cosine))
        temp_final = sorted(temp, key=lambda x: x[2], reverse=True)[:topk]
        temp1 = np.array([v3[0] for v1, v2, v3 in temp_final])

        # 计算出得相似度会出现nan情况，将值替换为均值
        if sum(np.isnan(temp1)) > 0:
            avg = np.mean(~np.isnan(temp1))
            temp_name, temp_long, temp_cos = [], [], []
            for t in temp_final:
                temp_name.append(t[0])
                temp_long.append(t[1])
                cos = avg if np.isnan(t[2]) else t[2]
                temp_cos.append(cos)

            final = sorted(list(zip(temp_name, temp_long, temp_cos)), key=lambda x: x[2], reverse=True)
        else:
            final = temp_final

        # 相似度为1的值过多，使用jaccard similarity更新相似度
        if len(temp1[temp1 == 1]) > topk/2:
            final_jaccard = []
            temp_name, temp_long, temp_cos = [], [], []
            query_jaccard = np.where(query[index] > 0, 1, 0).reshape((1, -1))[0]
            vector = np.zeros((2, len(query_jaccard)))
            vector[0, :] = query_jaccard
            for j in range(len(final)):
                # Jaccard similarity
                name_to_index = name_list.index(final[j][0])
                vector[1, :] = base_jaccard[name_to_index, :]
                join = np.sum(np.min(vector, axis=0))
                union = np.sum(np.max(vector, axis=0))
                res = 0.7 * join/union + 0.3 * final[j][2]
                temp_name.append(final[j][0])
                temp_long.append(final[j][1])
                temp_cos.append(res[0])

            final = sorted(list(zip(temp_name, temp_long, temp_cos)), key=lambda x: x[2], reverse=True)

        for value in final:
            result_list.append({'A': str(tr_final.loc[i, 'official_account']), 'B': value[0], 'similar': float(value[2]), 'islong': value[1]})

    result = pd.DataFrame(result_list).groupby('A')
    end = datetime.now()
    print("time-consuming: ", end-start)
    logger.info("calculate similarity successful! time:"+str(end-start))
    return result


# 评估建模效果 top3 - 20
def evaluate_embedding(article_vec, official_account_vec, desc):
    # top3, top5, top10, top20
    sum_top3, sum_top5, sum_top10, sum_top20 = 0, 0, 0, 0
    base_norm = 0
    size = len(article_vec)

    if desc not in {'tfidf', 'keyword', 'doc2vec', 'lda'}:
        return

    if desc in {'doc2vec', 'tfidf', 'keyword'}:
        base = np.array(list(official_account_vec['embedding']))
        base_temp = np.linalg.norm(base, axis=1).reshape((-1, 1))
        base_norm = base / base_temp

    for i in range(size):
        label = str(article_vec.loc[i, 'official_account'])
        if desc in {'doc2vec', 'tfidf', 'keyword'}:
            query = np.array(article_vec.loc[i, 'article_vector']).reshape((-1, 1))
            query_temp = np.linalg.norm(query)
            query_norm = query / query_temp
            cosine = list((base_norm @ query_norm))

        elif desc == 'lda':
            cosine = []
            query = article_vec.loc[i, 'article_vector']
            for i in range(len(official_account_vec)):
                res, base_list = 0, list(official_account_vec.loc[i, 'embedding'])
                for vec in base_list: res += cossim(query, vec)
                res /= len(base_list)
                cosine.append(res)

        else:
            pass

        temp = list(zip(official_account_vec['official_account'], cosine))
        final = sorted(temp, key=lambda x: x[1], reverse=True)

        for topk in [3, 5, 10, 20]:
            if label in set([i[0] for i in final[:topk]]):
                if topk == 3:
                    sum_top3 += 1
                if topk == 5:
                    sum_top5 += 1
                if topk == 10:
                    sum_top10 += 1
                if topk == 20:
                    sum_top20 += 1

    print("* number of official account: ", len(list(official_account_vec['embedding'])))
    print("* top3: ", sum_top3/size)
    print("* top5: ", sum_top5/size)
    print("* top10: ", sum_top10/size)
    print("* top20: ", sum_top20/size)
    print("* mean: ", (sum_top3/size + sum_top5/size + sum_top10/size + sum_top20/size)/4)


def evaluate_tfidf_lda_embedding(article_vec_tfidf, article_vec_lda, official_account_tfidf, official_account_lda, l1):
    base = np.array(list(official_account_tfidf['embedding']))
    base_temp = np.linalg.norm(base, axis=1).reshape((-1, 1))
    base_norm = base / base_temp
    sum_top3, sum_top5, sum_top10, sum_top20 = 0, 0, 0, 0
    size = len(article_vec_tfidf)

    for i in range(size):
        label = str(article_vec_tfidf.loc[i, 'official_account'])

        # tfidf: calculate cosine.
        query = np.array(article_vec_tfidf.loc[i, 'article_vector']).reshape((-1, 1))
        query_temp = np.linalg.norm(query)
        query_norm = query / query_temp
        cosine_tfidf = (base_norm @ query_norm).reshape((1, -1))

        # LDA: calculate cosine.
        cosine_lda = np.zeros(len(official_account_lda))
        query = article_vec_lda.loc[i, 'article_vector']
        for i in range(len(official_account_lda)):
            res, base_list = 0, list(official_account_lda.loc[i, 'embedding'])
            for vec in base_list: res += cossim(query, vec)
            res /= len(base_list)
            cosine_lda[i] = res

        # tfidf: 0.8, LDA: 0.2
        cosine_final = (l1*cosine_tfidf + (1-l1)*cosine_lda).tolist()[0]

        temp = list(zip(official_account_tfidf['official_account'], cosine_final))
        final = sorted(temp, key=lambda x: x[1], reverse=True)

        for topk in [3, 5, 10, 20]:
            if label in set([i[0] for i in final[:topk]]):
                if topk == 3:
                    sum_top3 += 1
                if topk == 5:
                    sum_top5 += 1
                if topk == 10:
                    sum_top10 += 1
                if topk == 20:
                    sum_top20 += 1

    print("* number of official account: ", len(list(official_account_tfidf['embedding'])))
    print("* top3: ", sum_top3 / size)
    print("* top5: ", sum_top5 / size)
    print("* top10: ", sum_top10 / size)
    print("* top20: ", sum_top20 / size)
    print("* mean: ", (sum_top3 / size + sum_top5 / size + sum_top10 / size + sum_top20 / size) / 4)


# 写数据库
def write_db(result, dict_stat):
    logger.info("start to update database...")
    print("start to update database...")
    start = datetime.now()
    args = []
    sql = "insert into kol_similar values(null,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    index = 1
    acc_set = set()
    connection = sql_init()
    with connection.cursor() as cursor:
        cursor.execute('truncate  table `kol_similar`')
        for account, sim_account in result:
                if str(account).lower() in acc_set:
                    logger.info("** repeated account: "+str(account))
                    continue

                # kol_similar 表对大小写不敏感，但是公众号名称区分大小写。
                acc_set.add(str(account).lower())
                sim_acc_set = set()
                sim_acc, value, islong = list(sim_account['B']), list(sim_account['similar']), list(sim_account['islong'])

                for i in range(len(sim_account)):

                    if str(sim_acc[i]) not in dict_stat:
                        logger.info('** stat not found. sim_acc: '+str(sim_acc[i]))
                        continue

                    if str(value[i]) == 'nan':
                        logger.info('** cosine is nan.')

                    if str(sim_acc[i]).lower() in sim_acc_set:
                        logger.info("** sim_acc repeat. account: "+str(account)+", sim_acc: "+str(sim_acc[i]))
                        continue

                    sim_acc_set.add(str(sim_acc[i]).lower())
                    inf = dict_stat[sim_acc[i]]
                    args.append((str(account), str(sim_acc[i]), inf[0], inf[1], inf[2], inf[3], int(islong[i]), float(value[i]), inf[4], inf[5], inf[6]))
                    # 批量写入数据库。
                    try:
                        if len(args) == 300:
                            logger.info("writing...")
                            cursor.executemany(sql, args)
                            connection.commit()
                            args.clear()
                            index += 1
                    except Exception as e:
                        args.clear()
                        logger.info(str(e))
                        pass
                    continue

    logger.info("update database time: "+str(datetime.now() - start))
    print("time consuming: ", datetime.now() - start)


def main():
    # 读库
    df, dict_stat = read_db()

    # 中文预处理
    logger.info("start to pre-processing...")
    print("start to pre-processing...")
    df["temp"] = df.apply(processing, axis=1)

    df[["words", "defined_keywords"]] = df["temp"].apply(pd.Series)

    # 计算tfidf
    tf, tfidf, vocab = calculate_tfidf(df)

    # 获得关键字的索引用于计算 sparse matrix
    temp_vocab = {v: k for k, v in vocab.items()}
    defined_keywords_dict = dict()
    for v in defined_keywords_set:
        if v in temp_vocab:
            index = temp_vocab[v]
            defined_keywords_dict[v] = index
        else:
            pass

    print("defined keywords: ")
    print(defined_keywords_dict)

    # 计算文章向量以及公众号向量 dense
    article_tfidf, official_account_tfidf = calculate_official_account_dense_tfidf(tfidf, df)

    result = calculate_cosine_dense_matrix_multiple(official_account_tfidf, defined_keywords_dict, None)

    # 写库
    write_db(result, dict_stat)


if __name__ == "__main__":
    main()