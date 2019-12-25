"""
FPGrowth 算法
Run with:
  bin/spark-submit --py-files='/Users/t/python/spark-learning/src/utils.zip' \
  /Users/t/python/spark-learning/src/ml/recommendation/fp_growth.py
"""
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("FPGrowth")\
        .getOrCreate()


    df = spark.createDataFrame([
        (0, [1, 2, 5]),
        (1, [1, 2, 3, 5]),
        (2, [1, 2])
    ], ["id", "items"])

    fpGrowth = FPGrowth(itemsCol="items", minSupport=0.5, minConfidence=0.6)
    model = fpGrowth.fit(df)

    model.freqItemsets.show()

    model.associationRules.show()

    model.transform(df).show()

    spark.stop()