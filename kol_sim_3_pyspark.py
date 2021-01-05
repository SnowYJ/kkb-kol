"""
方法2.1：pyspark version
"""
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
from pyspark.ml.feature import Word2Vec
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import IDF
from scipy.spatial import distance
import pymysql.cursors
from datetime import datetime
from pyspark.ml.linalg import Vectors
import csv
from pyspark.ml.feature import BucketedRandomProjectionLSH, MinHashLSH
from pyspark.sql import Row
from pyspark.sql.functions import col
import os
import logging

# os.environ['PYSPARK_PYTHON']='/usr/local/bin/python3'

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='kol_sim_3.log',
                    filemode='w')

logger = logging.getLogger(__name__)


# 自定义关键词字典，避免自动提取关键词时，对这些词的忽略。
defined_keywords_set = {'python', 'java', 'C', 'C++', 'web', 'CV', 'NLP', '运营', '算法'}


# 数据库连接
# def sql_init():
#     connection = pymysql.connect(host='192.168.100.54',
#                                  user='test',
#                                  password='Meihao100@bfbd',
#                                  database='kol',
#                                  charset='utf8mb4'
#                                  )
#     return connection
def sql_init():
    connection = pymysql.connect(host='192.168.30.200',
                                 user='root',
                                 password='qg1xzGZSZt',
                                 database='kol',
                                 charset='utf8mb4'
                                 )
    return connection


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
        if w in defined_keywords_set: is_keyword.add(w)
        res.append(w)
    return (" ".join(res), ",".join(list(is_keyword)))


# 读数据库（返回值：分词前的df，以及公众号统计信息）
def read_db():
    df = pd.DataFrame(columns=('official_account', 'longorshort', 'article'))
    dict_stat = dict()

    connection = sql_init()
    with connection.cursor() as cursor:
        sql_select = "select c.content, a.official_account_name from kol_content as c inner join kol_article as a on c.third_id = a.third_id where a.type_id = 1 LIMIT 60000"
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

    return (df, dict_stat)


# 根据tfidf获得文章关键词
def sort_by_tfidf(partition):
    topk, index = 20, 0
    for row in partition:
        _dict = list(zip(row.tfidf.indices, row.tfidf.values))
        _dict = sorted(_dict, key=lambda x: x[1], reverse=True)
        result = _dict[:topk]
        for word_index, tfidf in result:
            yield row.official_account, row.longorshort, index, row.article, int(word_index), round(float(tfidf), 4), row.defined_keywords
        index += 1


# 关键词用索引关联
def append_index(data):
    for index in range(len(data)):
        data[index] = list(data[index])  # convert tuple to list
        data[index].append(index)
        data[index][1] = float(data[index][1])


# 根据tfidf对词向量加权
def weighted_vector(row):
    return row.official_account, row.longorshort, row.keyword, row.tfidf * row.vector, row.defined_keywords


# 计算词向量
def calculate_word2vec(vector_size, sd):
    logger.info("start to calculate word embedding...")
    print("start to calculate word embedding...")
    word2vec = Word2Vec(vectorSize=vector_size, minCount=300, inputCol="words_splits", outputCol="word2vec")
    model = word2vec.fit(sd)
    word_vector = model.getVectors()

    # 提取自定义关键词的 embedding。
    defined_keyword_vector = word_vector.filter(col("word").isin(defined_keywords_set)).collect()
    defined_keyword_vector_dict = dict()
    for r in defined_keyword_vector:
        key, value = r['word'], r['vector']
        defined_keyword_vector_dict[key] = value

    return (word_vector, defined_keyword_vector_dict)

# 计算tfidf
def calculate_tfidf(sd):
    logger.info("start to calculate tfidf...")
    print("start to calculate tfidf...")
    cv = CountVectorizer(inputCol="words_splits", outputCol="tf", vocabSize=7000, minDF=5.0)
    cv_model = cv.fit(sd)
    tf = cv_model.transform(sd)
    idf = IDF(inputCol="tf", outputCol="tfidf")
    idf_model = idf.fit(tf)
    tfidf = idf_model.transform(tf)
    return (cv_model, idf_model, tfidf)


# 计算相似度
def calculate_lsh(tr_final):
    logger.info("start to calculate similarity...")
    print("start to calculate similarity...")
    start = datetime.now()
    brp = BucketedRandomProjectionLSH(inputCol='embedding', outputCol='hashes', numHashTables=4.0, bucketLength=10.0)
    model = brp.fit(tr_final)
    similar = model.approxSimilarityJoin(tr_final, tr_final, 2.0, distCol='EuclideanDistance')  # JaccardDistance
    result_temp = similar.select(
        col("datasetA.official_account").alias("A"),
        col("datasetB.official_account").alias("B"),
        col("datasetB.longorshort").alias('islong'),
        col("EuclideanDistance").alias("similar")
    )
    result = result_temp.toPandas().groupby('A')
    print("time consuming: ", datetime.now() - start)
    return result


# 写数据库
def write_db(result, dict_stat):
    logger.info("start to update database...")
    print("start to update database...")
    start = datetime.now()
    connection = sql_init()
    args = []
    sql = "insert into kol_similar values(null,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    index = 1
    try:
        connection = sql_init()
        with connection.cursor() as cursor:
            cursor.execute('truncate  table `kol_similar`')
            for account, sim_account in result:
                logger.info(account)
                sim_acc, value, islong = list(sim_account['B']), list(sim_account['similar']), list(sim_account['islong'])
                for i in range(len(sim_account)):
                    inf = dict_stat[sim_acc[i]]
                    args.append((account, sim_acc[i], inf[0], inf[1], inf[2], inf[3], islong[i], value[i], inf[4], inf[5], inf[6]))
                    # 批量写入数据库。
                    if len(args) == 500:
                        logger.info("####### writing...")
                        cursor.executemany(sql, args)
                        logger.info("####### writing successful!")
                        connection.commit()
                        args.clear()
                        index += 1
    finally:
        # cursor.executemany(sql, args)
        # connection.commit()
        connection.close()
    logger.info("successful, time: "+str(datetime.now() - start))
    print("time consuming: ", datetime.now() - start)


# 计算公众号向量
def calculate_official_account_embedding(sc, tfidf, word_vector, cv_model, idf_model, defined_keyword_vector_dict):
    """
    根据tfidf获取每个文章的前n个关键词，对关键词的word2vec乘tfidf，然后求平均作为文章的embedding。对文章的embedding取均值作为公众号的embedding。
    """
    # 对tfidf排序，获得每篇文章20个关键字的索引以及对应的tfidf值。
    keywords_by_tfidf = tfidf.rdd.mapPartitions(sort_by_tfidf)\
        .toDF(["official_account", "longorshort", "article_index", "article", 'keyword_index', 'tfidf', 'defined_keywords'])

    # 将关键词与索引关联。
    keywords_list_with_idf = list(zip(cv_model.vocabulary, idf_model.idf.toArray()))
    append_index(keywords_list_with_idf)
    keywords_index_df = sc.parallelize(keywords_list_with_idf).toDF(["keyword", "idf", "keyword_index"])

    # 将关键词与tfidf关联。将关键词与word2vec关联。# 这里改动
    result_df = keywords_by_tfidf\
        .join(keywords_index_df, keywords_by_tfidf.keyword_index == keywords_index_df.keyword_index, 'inner')\
        .join(word_vector, keywords_index_df.keyword == word_vector.word, 'inner')


    # 对关键词进行加权。
    official_account_keyword_vectors = result_df.select('official_account', 'keyword', 'longorshort', 'tfidf', 'vector', 'defined_keywords')\
        .rdd.map(weighted_vector).toDF(['official_account', 'longorshort', 'keyword', 'weighted_vector', 'defined_keywords'])

    # 根据公众号对文章进行分组。
    grouped = official_account_keyword_vectors\
        .select('official_account', 'longorshort', 'keyword', 'weighted_vector', 'defined_keywords')\
        .toPandas().groupby(['official_account'])

    # 计算公众号向量。
    logging.info("start to calculate official account embedding...")
    kol_vec = pd.DataFrame(columns=('official_account', 'longorshort', 'embedding'))
    for official_account, doc2vec in grouped:
        feature = list(doc2vec['weighted_vector'])
        keyword = set(doc2vec['keyword'])
        tag = list(doc2vec['longorshort'])

        # 提取公众号的自定义关键词，并与tfidf提取的关键词集合取补集，用来表示未出现的自定义关键词。
        defined_keyword = set(doc2vec['defined_keywords'].tolist())
        defined_key_list = []
        for s in defined_keyword:
            defined_key_list += s.split(",")
        temp = list(set(defined_key_list).difference(keyword))

        # 将自定义关键词的词向量加入feature中用于求平均。
        for k in temp:
            if k not in defined_keyword_vector_dict:
                continue
            feature.append(np.array(defined_keyword_vector_dict[k]))

        islong = False
        for i in tag:
            if not islong:
                islong = islong or i
            else:
                break

        kol_vec = kol_vec.append({'official_account': official_account, 'longorshort': islong, 'embedding': np.mean(feature, 0)}, ignore_index=True)

    return kol_vec


def main():
    # spark设置
    conf = (SparkConf().set("spark.cores.max", "16").set("spark.driver.memory", "16g")
            .set("spark.executor.memory", "16g").set("spark.executor.memory_overhead", "16g")
            .set("spark.driver.maxResultsSize", "0")
            )
    # .set("spark.rpc.message.maxSize", "1024")

    sc = SparkContext(appName="testRDD", conf=conf)
    spark = SparkSession(sc)

    # 读库
    df, dict_stat = read_db()

    # 中文预处理
    logger.info("start to pre-processing...")
    print("start to pre-processing...")
    df["temp"] = df.apply(processing, axis=1)
    df[["words", "defined_keywords"]] = df["temp"].apply(pd.Series)
    sd = spark.createDataFrame(df)
    sd = sd.withColumn('words_splits', F.split(sd.words, " "))

    # 计算 word embedding（长度100）
    vector_size = 100
    word_vector, defined_keyword_vector_dict = calculate_word2vec(vector_size, sd)

    # 计算 tfidf
    cv_model, idf_model, tfidf = calculate_tfidf(sd)

    # 计算公众号向量
    kol_vec = calculate_official_account_embedding(sc, tfidf, word_vector, cv_model, idf_model, defined_keyword_vector_dict)

    # 将kol_vec(pandas.dataframe) 转换成 spark.dataframe
    temp = pd.DataFrame(np.array(list(kol_vec['embedding'])).reshape(-1, vector_size))
    temp['official_account'] = kol_vec['official_account']
    temp['longorshort'] = kol_vec['longorshort']
    train = spark.createDataFrame(temp)
    # 将词向量由 numpy array 转换成 spark dense vector
    tr = train.rdd.map(lambda p: Row(official_account=p[vector_size], longorshort=p[vector_size+1], embedding=Vectors.dense(p[0:vector_size])))
    tr_final = spark.createDataFrame(tr)

    # 使用 LSH 计算公众号向量相似度
    result = calculate_lsh(tr_final)

    # 写库
    write_db(result, dict_stat)


if __name__ == "__main__":
    main()