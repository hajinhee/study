import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        shuffle= True, random_state=66, train_size = 0.8)

#2. 모델 구성
# model = GridSearchCV(SVC(), parameters, cv = kfold, verbose = 3,
#                      refit=True)   # GridSearchCV(모델, 파라미터, cv = 크로스 발리데이션)
model = SVC(C = 1, kernel ='linear', degree = 3)
# refit을 True로하면 가장 좋은 값을 

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

aaa = model.score(x_test, y_test)      # evaluate 개념
print('model.score :', model.score(x_test, y_test))

y_pred = model.predict(x_test) 
print('accuracy_score :', accuracy_score(y_test, y_pred))