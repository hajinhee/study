from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=100)

#2. 모델
model = CatBoostClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('CatBoostClassifier :', result)
print('accuracy_score :', acc)


'''
CatBoostClassifier : 1.0
Perceptron : 0.6111111111111112
LinearSVC : 0.9722222222222222
SVC : 0.5555555555555556
KNeighborsClassifier : 0.6388888888888888
LogisticRegression : 0.9444444444444444
DecisionTreeClassifier : 0.75
RandomForestClassifier : 1.0
XGBClassifier : 0.8888888888888888
LGBMClassifier : 0.9722222222222222
CatBoostClassifier : 1.0
'''