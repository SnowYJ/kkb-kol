import os
from enum import Enum


class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    FIRST_TOKEN = 4  # 对应[CLS]，对单个序列
    LAST_TOKEN = 5  # 对应[SEP]
    CLS_TOKEN = 4  # 第一个token.
    SEP_TOKEN = 5  # 最后一个token

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()


file_path = os.path.dirname(os.path.dirname(__file__))

model_dir = os.path.join(file_path, 'chinese_L-12_H-768_A-12')
config_name = os.path.join(model_dir, 'bert_config.json')
ckpt_name = os.path.join(model_dir, 'bert_model.ckpt')
vocab_file = os.path.join(model_dir, 'vocab.txt')

xla = True
layer_indexes = [-2]
