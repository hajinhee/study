import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state=100)

model = SVC()

scores = cross_val_score(model, x, y, cv = kfold)
print('ACC :',scores, "\ncross_val_score :", round(np.mean(scores),4))


'''
ACC : [1.         1.         0.93333333 0.96666667 1.        ] 
5번 훈련 시켰다. 
평균값 : 0.9800000000000001
cross_val_score : 0.98
'''
