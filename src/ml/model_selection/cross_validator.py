"""
cross_validator
Run with:
  bin/spark-submit --py-files='/Users/t/python/spark-learning/src/utils.zip' \
  /Users/t/python/spark-learning/src/ml/model_selection/cross_validator.py
"""

from sklearn import datasets
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.sql import SparkSession

def f(x, y):

    rel = {}
    rel['features'] = Vectors.dense(float(x[0]), float(x[1]), float(x[2]), float(x[3]))
    rel['label'] = str(y)
    return rel

if __name__ == "__main__":
    spark = SparkSession.builder.appName("CrossValidator").getOrCreate()
    digits = datasets.load_iris()
    X, y = digits.data, digits.target

    row = Row()
    for i, p in enumerate(X):
        row += Row(f(p, y[i]))

    data = spark.createDataFrame(row)

    train, test = data.randomSplit([0.7, 0.3])

    labelIndexer = StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
    featureIndexer = VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(data)

    lr = LogisticRegression().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(50)
    labelConverter = IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(
        labelIndexer.labels)
    lrPipeline = Pipeline().setStages([labelIndexer, featureIndexer, lr, labelConverter])

    # 使用ParamGridBuilder方便构造参数网格. 其中regParam参数定义规范化项的权重；elasticNetParam是Elastic net 参数，取值介于0和1之间。
    # elasticNetParam设置2个值，regParam设置3个值。最终将有(3 * 2) = 6个不同的模型将被训练。
    paramGrid = ParamGridBuilder().addGrid(lr.elasticNetParam, [0.2, 0.8]).addGrid(lr.regParam, [0.01, 0.1, 0.5]).build()

    # 对于回归问题评估器可选择RegressionEvaluator，二值数据可选择BinaryClassificationEvaluator，多分类问题可选择MulticlassClassificationEvaluator
    cv = CrossValidator().setEstimator(lrPipeline).setEvaluator(
        MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol(
            "prediction")).setEstimatorParamMaps(paramGrid).setNumFolds(3)
    cvModel = cv.fit(train)

    lrPredictions = cvModel.transform(test)
    lrPreRel = lrPredictions.select("predictedLabel", "label", "features", "probability").collect()
    for item in lrPreRel:
        print(str(item['label']) + ',' + str(item['features']) + '-->prob=' + str(
            item['probability']) + ',predictedLabel' + str(item['predictedLabel']))
    evaluator = MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction")
    lrAccuracy = evaluator.evaluate(lrPredictions)
    print("lrAccuracy:{}".format(lrAccuracy))

    # 获取最优的逻辑斯蒂回归模型，并查看其具体的参数
    bestModel = cvModel.bestModel
    lrModel = bestModel.stages[2]
    print("Coefficients: " + str(lrModel.coefficientMatrix) + "Intercept: " + str(
        lrModel.interceptVector) + "numClasses: " + str(lrModel.numClasses) + "numFeatures: " + str(
        lrModel.numFeatures))

    print(lr.explainParam(lr.regParam))
    print(lr.explainParam(lr.elasticNetParam))
