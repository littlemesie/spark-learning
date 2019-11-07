"""
kmeans算法
Run with:
  bin/spark-submit --py-files='/Users/t/python/spark-learning/src/utils.zip' \
  /Users/t/python/spark-learning/src/ml/kmeans.py
"""
from pyspark.ml.clustering import KMeans,KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from utils import constants

def f(x):
    rel = {}
    rel['features'] = Vectors.dense(float(x[0]), float(x[1]))
    return rel

def load_data(sc, path=None):
    """加载数据"""
    data = sc.textFile(path).map(lambda line: line.split('\t')).map(lambda p: Row(**f(p))).toDF()

    return data

if __name__ == "__main__":
    spark = SparkSession.builder.appName("KMeans").getOrCreate()

    sc = spark.sparkContext
    # 加载数据
    dataset = load_data(sc, constants.DATA_PATH + '10.KMeans/testSet.txt')

    # 训练模型
    # 也可以KMeans().setK(2).setSeed(1).setFeaturesCol('features').setPredictionCol('prediction')
    kmeans = KMeans(featuresCol="features", predictionCol="prediction", k=2, seed=1)
    model = kmeans.fit(dataset)

    # 结果
    predictions = model.transform(dataset)
    results = predictions.collect()
    for item in results:
        print(str(item[0]) + 'is predcted as cluster' + str(item[1]))

    # 通过计算Silhouette得分评估聚类
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    print("Silhouette with squared euclidean distance = " + str(silhouette))

    # 获取到模型的所有聚类中心情况
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    spark.stop()