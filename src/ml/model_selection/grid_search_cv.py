"""
调参
Run with:
  bin/spark-submit --py-files='/Users/t/python/spark-learning/src/utils.zip' \
  /Users/t/python/spark-learning/src/ml/model_selection/grid_search_cv.py
"""
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
from spark_sklearn.util import createLocalSparkSession
from spark_sklearn.grid_search import GridSearchCV
digits = datasets.load_digits()
X, y = digits.data, digits.target

sc = createLocalSparkSession().sparkContext
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [0.1, 0.2, 0.3],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"],
              "n_estimators": [10, 20, 40, 80]}

gs = GridSearchCV(sc, RandomForestClassifier(), param_grid=param_grid)
gs.fit(X, y)

# 获取最佳参数
best_params_ = None
best_score_ = 0
params = gs.cv_results_['params']
mean_train_score = gs.cv_results_['mean_train_score']
for i, score in enumerate(mean_train_score):
    if i == 0:
        best_score_ = score
        best_params_ = params[i]
    if score > best_score_:
        best_score_ = score
        best_params_ = params[i]
print(best_params_)
print(best_score_)

