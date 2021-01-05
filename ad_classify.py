import pymysql.cursors
from keras.models import load_model
from bert.extract_feature import BertVector
import numpy as np
from dbutils.pooled_db import PooledDB
import logging


class ad_classify(object):

    def __init__(self, db_host, db_user, db_password, db_name, db_port=3306, step=100):
        self.pool = PooledDB(
            creator=pymysql,  # 使用链接数据库的模块
            maxconnections=6,  # 连接池允许的最大连接数，0和None表示不限制连接数
            mincached=2,  # 初始化时，链接池中至少创建的空闲的链接，0表示不创建
            maxcached=5,  # 链接池中最多闲置的链接，0和None不限制
            maxshared=3,
            # 链接池中最多共享的链接数量，0和None表示全部共享。PS: 无用，因为pymysql和MySQLdb等模块的 threadsafety都为1，所有值无论设置为多少，_maxcached永远为0，所以永远是所有链接都共享。
            blocking=True,  # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
            maxusage=None,  # 一个链接最多被重复使用的次数，None表示无限制
            setsession=[],  # 开始会话前执行的命令列表。
            ping=0,
            # ping MySQL服务端，检查是否服务可用。
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
            charset='utf8mb4'
        )

        self.step = step
        self.load_model = load_model("visit_classify.h5")
        self.bert_model = BertVector(pooling_strategy="REDUCE_MEAN", max_seq_len=100)

    def query(self):
        while True:
            sql = f"SELECT a.third_id, b.content FROM kol_article a LEFT JOIN kol_content b on a.third_id = b.third_id LIMIT {self.step}"# where a.type_id is null

            arr = []
            connection = self.pool.connection()
            cursor = connection.cursor()
            cursor.execute(sql)
            for x in cursor.fetchall():
                third_id, content = x
                y = 0
                if content:
                    vec = self.bert_model.encode([content[len(str(content)) - 300:]])["encodes"][0]
                    x_train = np.array([vec])
                    predicted = self.load_model.predict(x_train)
                    y = np.argmax(predicted[0])
                arr.append((int(y), third_id))
            if arr:
                yield arr
            else:
                break

    def update_type_id(self):
        sql = "update kol_article a set a.type_id = %s where a.third_id = %s"
        connection = self.pool.connection()
        cursor = connection.cursor()
        for x in self.query():
            len_str = str(len(x))
            cursor.executemany(sql, x)
            connection.commit()
            logging.info(f'更新了 {len_str} 条type id')

    def __del__(self):
        self.pool.close()


if __name__ == '__main__':
    ac = ad_classify(
        db_host='192.168.2.157',
        db_user='root',
        db_password='8Df#gnoYrmvIhxA920',
        db_name='kol'
    )
    ac.update_type_id()