"""数据加载"""
import pandas as pd
from collections import Counter

dataset_df = pd.read_excel('data/data.xlsx')[['label', 'content']]
dataset_df.rename(columns={'content': 'text'}, inplace=True)

print(dataset_df.head(3))
dataset_df['text'] = dataset_df['text'].apply(lambda x: x[-300:])
print('keep last 300: ')
print(dataset_df.head(3))

train_df = dataset_df.iloc[:2000]
test_df = dataset_df.iloc[2000:]

print(train_df.shape)
print(test_df.shape)
print('label distribution：', Counter(train_df['label']))





# ==================================
# def read_txt_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         content = [_.strip() for _ in f.readlines()]
#
#     labels, texts = [], []
#     for line in content:
#         parts = line.split()
#         label, text = parts[0], ''.join(parts[1:])
#         labels.append(label)
#         texts.append(text)
#     return labels, texts
#
# file_path = 'data/train.txt'
# labels, texts = read_txt_file(file_path)
# train_df = pd.DataFrame({'label': labels, 'text': texts})
#
# file_path = 'data/test.txt'
# labels, texts = read_txt_file(file_path)
# test_df = pd.DataFrame({'label': labels, 'text': texts})
#
# print(train_df.head())
# print(test_df.head())
#
# train_df['text_len'] = train_df['text'].apply(lambda x: len(x))
# print(train_df.describe())

