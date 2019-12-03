"""
 word2vec 算法
 bin/spark-submit --py-files='/Users/t/python/spark-learning/src/utils.zip' \
  /Users/t/python/spark-learning/src/ml/nlp/word2vec.py
"""
from pyspark.ml.feature import Word2Vec
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Word2Vec").getOrCreate()


    documentDF = spark.createDataFrame([
        ("Hi I heard about Spark".split(" "), ),
        ("I wish Java could use case classes".split(" "), ),
        ("Logistic regression models are neat".split(" "), )
    ], ["text"])


    word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
    print(word2Vec)
    model = word2Vec.fit(documentDF)

    result = model.transform(documentDF)
    print(result)
    for row in result.collect():
        text, vector = row
        print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))


    spark.stop()