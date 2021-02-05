### kol任务介绍
    1. 公众号分类任务，判断其是否属于广告公众号（评判方法：如果公众号下的文章包含广告即为广告公众号）。
    2. 公众号相似度匹配任务（判断公众号两两之间的相似度）。


<img src="experiment result/interface.PNG">

### 预训练模型
    Bert: 下载bert已训练好的模型文件 -- chinese_L-12_H-768_A-12
    trained-word2vec-model: 训练好的词向量
    trained-LDA-model: 训练好的LDA模型

### 方法
    分类模型：Bert+MLP
    公众号向量：tfidf, doc2vec, keywords embedding, LDA, tfidf+LDA, BM25
    相似度度量：cosine 或 cosine + Jaccard（避免相似度为1的情况过多）
    相似度计算：MIPS, LSH
    内存优化策略: Inverted Index, sparse matrix compress
    
### 模型效果
    1. Bert分类模型的accuracy, recall, precision均在85%以上。
    2. 在文章数量10000公众号数量6101情况下，tfidf+LDA的建模效果最好。
    3. 相似度计算速度MIPS快于LSH（MIPS：0:06:49 LSH: 0:54:15）。
    4. 数据量增加建模效果越好。
    (参考 experiment result/kol-similar.pages)

### 代码文件说明
    1. load_data.py: 训练数据加载（训练集：data/data.xlsx 2862条数据，1为广告，0非广告）
    2. model_train.py: 使用keras训练Bert分类模型
    3. train_word2vec.py: 使用gensim训练词向量
    4. ad_classify.py: 给数据打标签
    5. kol_sim_3_python0.py: python版本相似度计算（每一次更新全部数据）
    6. kol_sim_3_python1.py: python版本相似度计算（使用Inverted Index优化内存使用）（每一次添加新公众号以及更新旧数据）
    7. kol_sim_spark.py: spark版本相似度计算

### 其余文件说明
    1. inverted_index_vocab.npy: 倒排索引字典
    2. trained-LDA-model: LDA预训练模型
    3. trained-word2vec-model: word2vec预训练模型
    4. test_corpus-temp.txt: 80万篇文章用于词向量训练
    5. data: 分类模型的训练数据
    6. visit_classify.h5: 分类模型

### 环境依赖
    python部分：python3 -m pip install -r ./requirements.txt -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
    spark部分：安装openjdk，配置java环境

### 数据库说明
    kol库下四个表：kol_similar, kol_content, kol_article, kol_statistics
    1. kol_article: 公众号基本信息（公众号ID，公众号名称，是否属于广告公众号）
    2. kol_content: 文章内容
    3. kol_similar: 相似度表
    4. kol_statistics: 公众号的统计信息（如：点赞量等）

### 参考
    BM25参考：https://my.oschina.net/stanleysun/blog/1617727 
    关键字：https://blog.csdn.net/qq_29153321/article/details/104680282