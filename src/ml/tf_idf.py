"""
tf-idf 算法
Run with:
  bin/spark-submit --py-files='/Users/t/python/spark-learning/src/utils.zip' \
  /Users/t/python/spark-learning/src/ml/tf_idf.py
"""
from __future__ import print_function

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession



if __name__ == "__main__":

    spark = SparkSession.builder.appName("tf-idf").getOrCreate()

    # $example on$
    sentenceData = spark.createDataFrame([
        (0.0, "Hi I heard about Spark"),
        (0.0, "I wish Java could use case classes"),
        (1.0, "Logistic regression models are neat")
    ], ["label", "sentence"])

    # Tokenizer把句子划分为单个词语
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData = tokenizer.transform(sentenceData)

    """
    TF: HashingTF 是一个Transformer，在文本处理中，接收词条的集合然后把这些集合转化成固定长度的特征向量。这个算法在哈希的同时会统计各个词条的词频。
    numFeatures: hash表的桶数
    """
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    featurizedData = hashingTF.transform(wordsData)

    """
    IDF: IDF是一个Estimator，在一个数据集上应用它的fit（）方法，产生一个IDFModel。 该IDFModel 接收特征向量（由HashingTF产生），
    然后计算每一个词在文档中出现的频次。IDF会减少那些在语料库中出现频率较高的词的权重。
    """
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    rescaledData.select("label", "features").show()

    spark.stop()
