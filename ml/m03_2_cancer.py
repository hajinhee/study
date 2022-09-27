from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np


#1. 데이터
datasets = load_breast_cancer()
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
Perceptron : 0.9473684210526315
LinearSVC : 0.8245614035087719
SVC : 0.9473684210526315
KNeighborsClassifier : 0.9473684210526315
LogisticRegression : 0.956140350877193
DecisionTreeClassifier : 0.956140350877193
RandomForestClassifier : 0.9736842105263158
XGBClassifier : 0.956140350877193
LGBMClassifier : 0.956140350877193
CatBoostClassifier : 0.9649122807017544
CatBoostClassifier : 0.9649122807017544
'''
