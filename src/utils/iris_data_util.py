from sklearn import datasets
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.sql import SparkSession
"""
将sklearn的iris转成spark的输入

"""
def f(x, y):
    ret = {}
    ret['features'] = Vectors.dense(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
    ret['label'] = str(y)
    return ret

def get_iris_dataframe(spark):
    digits = datasets.load_iris()
    X, y = digits.data, digits.target
    row = Row()
    for i, p in enumerate(X):
        row += Row(f(p, y[i]))

    data = spark.createDataFrame(row)

    # train, test = data.randomSplit([0.7, 0.3])
    return data
