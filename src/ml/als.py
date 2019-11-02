"""
als 算法
Run with:
  bin/spark-submit --py-files='/Users/t/python/spark-learning/src/utils.zip' \
  /Users/t/python/spark-learning/src/ml/als.py
"""
from __future__ import print_function
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from utils import constants


def load_data(spark, path=None):
    """加载数据"""
    lines = spark.read.text(path).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratings_rdd = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=int(p[3])))
    ratings = spark.createDataFrame(ratings_rdd)

    return ratings

if __name__ == "__main__":

    spark = SparkSession.builder.appName("als").getOrCreate()
    path = constants.DATA_PATH + '16.RecommenderSystems/ml-1m/ratings.dat'

    # 加载数据
    ratings = load_data(spark, path)
    train, test = ratings.randomSplit([0.8, 0.2])


    # Build the recommendation model using ALS on the train data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    als = ALS(maxIter=10, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")

    model = als.fit(train)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    # Generate top 10 movie recommendations for each user
    user_recs = model.recommendForAllUsers(10)
    recs = user_recs.select("userId", "recommendations").collect()
    for ur in recs:
        recommendations = list(row['movieId'] for row in ur['recommendations'])
        print("userId: {}, recommendations: {}".format(ur['userId'], ur['recommendations']))
    # # Generate top 10 user recommendations for each movie
    # movieRecs = model.recommendForAllItems(10)
    #
    # # Generate top 10 movie recommendations for a specified set of users
    # users = ratings.select(als.getUserCol()).distinct().limit(3)
    # userSubsetRecs = model.recommendForUserSubset(users, 10)
    # # Generate top 10 user recommendations for a specified set of movies
    # movies = ratings.select(als.getItemCol()).distinct().limit(3)
    # movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    # # $example off$
    # userRecs.show()
    # movieRecs.show()
    # userSubsetRecs.show()
    # movieSubSetRecs.show()

    spark.stop()
