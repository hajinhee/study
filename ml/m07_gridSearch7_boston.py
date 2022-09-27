import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x,y,
        shuffle= True, random_state=66, train_size = 0.8)
n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state=100)

parameters = [
    {'n_estimators': [100,200],'max_depth' : [6,8,10, 12],
    'min_samples_leaf' :[3,5,7,10],'min_samples_split' : [2,3,5,10]}]

#2. 모델 구성
model = GridSearchCV(RandomForestRegressor(), parameters, cv = kfold, verbose = 3,
                     refit=True, n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print('최적의 매개 변수 :', model.best_estimator_)
print('최적의 파라미터 : ', model.best_params_)
print('best_score_ :', model.best_score_)
aaa = model.score(x_test, y_test)      # evaluate 개념
print('model.score :', model.score(x_test, y_test))
y_pred = model.predict(x_test) 
print('accuracy_score :', accuracy_score(y_test, y_pred))
# y_pred_best = model.best_estimator_.predict(x_test)       ### 권장한다.
# print('최적 튠 ACC :', accuracy_score(y_test, y_pred_best))

'''
최적의 매개 변수 : RandomForestRegressor(max_depth=10, min_samples_leaf=3, min_samples_split=3)
최적의 파라미터 :  {'max_depth': 10, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 100}
best_score_ : 0.8186732753774212
model.score : 0.9220364076774521
'''