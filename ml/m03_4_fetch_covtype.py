from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=100)

#2. 모델
model = SVC()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('SVC :', result)
print('accuracy_score :', acc)

'''
CatBoostClassifier : 0.8840047158851321
LGBMClassifier : 0.8481020283469446
XGBClassifier : 0.8669655688751581
RandomForestClassifier : 0.9548118379043571
DecisionTreeClassifier : 0.9395884787828197
LogisticRegression : 0.620767105840641

'''