"""
更新库：判断文章是否为广告文
"""
import pymysql.cursors
from keras.models import load_model
from bert.extract_feature import BertVector
import numpy as np

"""数据库连接"""
query_connection = pymysql.connect(host='192.168.2.157',
                                   user='root',
                                   password='8Df#gnoYrmvIhxA920',
                                   database='kol',
                                   charset='utf8mb4',
                                   cursorclass=pymysql.cursors.SSCursor
                                   )

update_connection = pymysql.connect(host='192.168.2.157',
                                   user='root',
                                   password='8Df#gnoYrmvIhxA920',
                                   database='kol',
                                   charset='utf8mb4',
                                   cursorclass=pymysql.cursors.SSCursor
                                   )

"""模型加载"""
load_model = load_model("visit_classify.h5")
bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=100)

sql_select = "select c.third_id, c.content from kol_content as c inner join kol_article as a on c.third_id = a.third_id where a.type_id is null"
sql_update = "update kol_article a set a.type_id = %s where a.third_id = %s"


"""操作数据库"""

args = []


def query():
    try:
        with query_connection.cursor() as cursor:
            cursor.execute(sql_select)
            result = cursor.fetchone()
            while True:
                if result:
                    third_id = result[0]
                    content = str(result[1])
                    if content:
                        vec = bert_model.encode([content[len(str(content)) - 300:]])["encodes"][0]
                        x_train = np.array([vec])
                        predicted = load_model.predict(x_train)
                        y = np.argmax(predicted[0])
                    else:
                        y = 0
                    args.append((int(y), third_id))
                    print(len(args))
                    if len(args) == 500:
                        yield args
                    result = cursor.fetchone()
                else:
                    break

    finally:
        query_connection.close()


if __name__ == '__main__':
    try:
        with update_connection.cursor() as cursor:
            for args in query():
                cursor.executemany(sql_update, args)
                args.clear()
                update_connection.commit()
    finally:
        update_connection.close()