import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd


#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x,y,
        shuffle= True, random_state=66, train_size = 0.8)
n_splits = 3
kfold = KFold(n_splits = n_splits, shuffle = True, random_state=100)

parameters = [
    {'C':[1,10,100,1000,10000], 'kernel':['linear'], 'degree' : [3,4,5,6,7]},  # 25개
    {'C':[1, 10 ,100,1000], 'kernel':['rbf'], 'gamma':[0.001, 0.0001,0.01]},   # 12개
    {'C':[1, 10, 100, 1000,10000], 'kernel': ['sigmoid'], 
     'gamma':[0.01, 0.001, 0.0001], 'degree': [3,4,5,6,7]}                 # 75개
]                                                                    # 총 112번(더해준다) 

#2. 모델 구성
# model = GridSearchCV(SVC(), parameters, cv = kfold, verbose = 3,
#                      refit=True, n_jobs=1)   
model = RandomizedSearchCV(SVC(),parameters, cv = kfold, verbose = 3,
                     refit=True, n_jobs=1, n_iter=20)

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측

print('걸린 시간 :', end - start)
print('최적의 매개 변수 :', model.best_estimator_)
print('최적의 파라미터 : ', model.best_params_)

print('best_score_ :', model.best_score_)
aaa = model.score(x_test, y_test)      # evaluate 개념
print('model.score :', model.score(x_test, y_test))
y_pred = model.predict(x_test) 
print('accuracy_score :', accuracy_score(y_test, y_pred))
y_pred_best = model.best_estimator_.predict(x_test)       ### 권장한다.
print('최적 튠 ACC :', accuracy_score(y_test, y_pred_best))


'''
- GridSearch
걸린 시간 : 0.9189214706420898
최적의 매개 변수 : SVC(C=10, gamma=0.01)
최적의 파라미터 :  {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
best_score_ : 0.9833333333333332
model.score : 0.9666666666666667
accuracy_score : 0.9666666666666667
최적 튠 ACC : 0.9666666666666667

- randomSearch
걸린 시간 : 0.07557320594787598
최적의 매개 변수 : SVC(C=100, degree=7, gamma=0.001, kernel='sigmoid')
최적의 파라미터 :  {'kernel': 'sigmoid', 'gamma': 0.001, 'degree': 7, 'C': 100}
best_score_ : 0.9833333333333332
model.score : 1.0
accuracy_score : 1.0
최적 튠 ACC : 1.0

'''


