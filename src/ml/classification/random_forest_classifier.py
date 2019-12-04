"""
random forest 算法
Run with:
  bin/spark-submit --py-files='/Users/t/python/spark-learning/src/utils.zip' \
  /Users/t/python/spark-learning/src/ml/classification/random_forest_classifier.py
"""
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from utils import constants

def f(x):
    rel = {}
    if x[-1] == 'R':
        label = 0
    else:
        label = 1
    rel['features'] = Vectors.dense(list(map(lambda m: float(m), x[:-1])))
    rel['label'] = label
    return rel

def load_data(spark, path=None):
    """加载数据"""
    data = spark.sparkContext.textFile(constants.DATA_PATH + "7.RandomForest/sonar-all-data.txt")\
        .map(lambda line: line.split(',')).map(lambda p: Row(**f(p))).toDF()

    return data
if __name__ == "__main__":
    spark = SparkSession.builder.appName("RandomForestClassifier").getOrCreate()

    # Load training data
    data = load_data(spark)
    print(data.show(truncate=False))
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (train, test) = data.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(train)

    # Make predictions.
    predictions = model.transform(test)

    # Select example rows to display.
    print(predictions.select("predictedLabel", "label", "features").show(5, truncate=False))

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)

    spark.stop()