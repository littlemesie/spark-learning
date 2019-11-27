"""
logistic regression 算法
Run with:
  bin/spark-submit --py-files='/Users/t/python/spark-learning/src/utils.zip' \
  /Users/t/python/spark-learning/src/ml/classification/decision_tree_classification.py
"""
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from utils import iris_data_util


if __name__ == "__main__":

    spark = SparkSession.builder.appName("DecisionTree").getOrCreate()

    # Load training data
    data = iris_data_util.get_iris_dataframe(spark)
    train, test = data.randomSplit([0.7, 0.3])

    # 分别获取标签列和特征列，进行索引，并进行了重命名。
    labelIndexer = StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

    featureIndexer = VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

    # 这里我们设置一个labelConverter，目的是把预测的类别重新转化成字符型的。
    labelConverter = IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(
        labelIndexer.labels)
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(train)

    # Make predictions.
    predictions = model.transform(test)


    print(predictions.select("prediction", "indexedLabel", "features").show(5))

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g " % (1.0 - accuracy))

    treeModel = model.stages[2]

    print(treeModel)

    spark.stop()