"""
logistic regression 算法
Run with:
  bin/spark-submit --py-files='/Users/t/python/spark-learning/src/utils.zip' \
  /Users/t/python/spark-learning/src/ml/logistic_regression.py
"""
from __future__ import print_function
from pyspark.sql import Row, functions
from pyspark.ml.linalg import Vector, Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer,HashingTF, Tokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from utils import constants

def f(x):
    rel = {}
    rel['features'] = Vectors.dense(float(x[0]), float(x[1]))
    rel['label'] = int(x[2])
    return rel

def load_data(spark, path=None):
    """加载数据"""
    data = spark.sparkContext.textFile(constants.DATA_PATH + "5.Logistic/TestSet.txt")\
        .map(lambda line: line.split('\t')).map(lambda p: Row(**f(p))).toDF()

    return data

if __name__ == "__main__":

    spark = SparkSession.builder.appName("LogisticRegression").getOrCreate()

    # Load training data
    data = load_data(spark)

    # 切分训练集和数据集
    train, test = data.randomSplit([0.7, 0.3])

    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(train)

    # Extract the summary from the returned LogisticRegressionModel instance trained
    # in the earlier example
    trainingSummary = lrModel.summary

    # Obtain the objective per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)

    # Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    trainingSummary.roc.show()
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

    # Set the model threshold to maximize F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
    print("maxFMeasure:{}".format(maxFMeasure))
    bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
        .select('threshold').head()['threshold']
    print("bestThreshold:{}".format(bestThreshold))
    lr.setThreshold(bestThreshold)

    # 测试集上结果
    predict = lrModel.transform(test)
    print(predict)
    preRel = predict.select("features", "label", "prediction", "probability").collect()
    for p in preRel:
        print(str(p['features']) + ':' + str(p['label']) + '--> predict:' + str(p['prediction']) + ':' + str(p['probability']))

    spark.stop()