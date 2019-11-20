"""
GBTD Classifier 算法
Run with:
  bin/spark-submit --py-files='/Users/t/python/spark-learning/src/utils.zip' \
  /Users/t/python/spark-learning/src/ml/recommendation/gradient_boosted_tree_classifier.py
"""
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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

if __name__ == '__main__':
    spark = SparkSession.builder.appName("GBTD").getOrCreate()
    data = load_data(spark)
    # 切分训练集和数据集
    train, test = data.randomSplit([0.7, 0.3])

    string_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
    label_indexer = string_indexer.fit(data)

    # Set maxCategories so features with > 2 distinct values are treated as continuous.
    feature_indexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=2).fit(data)

    gbt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)

    # Chain indexers and GBT in a Pipeline
    pipeline = Pipeline(stages=[label_indexer, feature_indexer, gbt])

    # 训练模型
    model = pipeline.fit(train)

    # 测试
    predictions = model.transform(test)

    predictions.select("prediction", "indexedLabel", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    gbt_model = model.stages[2]
    print(gbt_model)

    spark.stop()