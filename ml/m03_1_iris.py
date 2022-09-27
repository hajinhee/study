import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


#1. 데이터

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 100)


#2. 모델구성
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
# 회귀처럼 보이지만, 분류모델에서 사용한다. // 이진 분류에 특화되어 있지만 다중분류에도 사용가능하다.
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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
Perceptron : 1.0
SVC : 1.0
KNeighborsClassifier : 1.0
LogisticRegression : 0.9666666666666667
DecisionTreeClassifier : 0.9666666666666667
RandomForestClassifier : 0.9666666666666667
XGBClassifier : 0.9666666666666667
LGBMClassifier : 0.9666666666666667
CatBoostClassifier : 0.9666666666666667
'''