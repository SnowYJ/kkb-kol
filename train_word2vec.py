"""
训练词向量
"""
import multiprocessing
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import pymysql.cursors
import logging
from gensim.models.word2vec import Text8Corpus

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='train_word2vec.log',
                    filemode='w')

logger = logging.getLogger(__name__)


def sql_init():
    connection = pymysql.connect(host='192.168.30.200',
                                 user='root',
                                 password='qg1xzGZSZt',
                                 database='kol',
                                 charset='utf8mb4'
                                 )
    return connection


def read_db():
    df = pd.DataFrame(columns=['article'])

    connection = sql_init()
    logger.info("start to read database...")
    with connection.cursor() as cursor:
        sql_select = "select c.content from kol_content as c LIMIT 800000"
        cursor.execute(sql_select)
        data = cursor.fetchall()
        for i in data:
            df = df.append({'article': str(i[0])}, ignore_index=True)
    logger.info("read database successful!")
    return df


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
        w = w.lower()
        res.append(w)
    return " ".join(res)


def main():
    df = read_db()
    corpus = 'test_corpus-temp.txt'
    logger.info("start to pre-processing...")
    print("start to pre-processing...")
    df["words"] = df.apply(processing, axis=1)
    df = df[df["words"] != 'none']
    df["words"].to_csv(corpus, index=False, header=None)

    logger.info("start to train word2vec...")
    print("start to train word2vec...")
    model = Word2Vec(Text8Corpus(corpus), size=100, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save('trained-word2vec-model/pretrained_word2vec.model')
    logger.info("model saved successful!")
    print("model saved successful!")


if __name__ == "__main__":
    main()