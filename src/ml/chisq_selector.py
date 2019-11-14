"""
卡方选择器 算法
Run with:
  bin/spark-submit --py-files='/Users/t/python/spark-learning/src/utils.zip' \
  /Users/t/python/spark-learning/src/ml/chisq_selector.py
"""


from pyspark.sql import SparkSession
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

if __name__ == "__main__":
    spark = SparkSession.builder.appName("ChiSqSelector").getOrCreate()


    df = spark.createDataFrame([
        (Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0,),
        (Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0,),
        (Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0,)], ["features", "clicked"])

    selector = ChiSqSelector(numTopFeatures=2, featuresCol="features",
                             outputCol="selectedFeatures", labelCol="clicked")

    result = selector.fit(df).transform(df)

    print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
    result.show()

    spark.stop()