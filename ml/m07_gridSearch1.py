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
n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state=100)

parameters = [
    {'C':[1,10,100,1000,10000], 'kernel':['linear'], 'degree' : [3,4,5,6,7]},  # 25개
    {'C':[1, 10 ,100,1000], 'kernel':['rbf'], 'gamma':[0.001, 0.0001,0.01]},   # 12개
    {'C':[1, 10, 100, 1000,10000], 'kernel': ['sigmoid'], 
     'gamma':[0.01, 0.001, 0.0001], 'degree': [3,4,5,6,7]}                 # 75개
]                                                                     # 총 112번(더해준다) 


#2. 모델 구성
model = GridSearchCV(SVC(), parameters, cv = kfold, verbose = 3,
                     refit=True, n_jobs=1)   # GridSearchCV(모델, 파라미터, cv = 크로스 발리데이션)
# model = SVC(c = 1, kernel ='linear', degree = 3)
# refit을 True로하면 가장 좋은 값을 
# n_jobs = cpu 갯수

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측

# x_test = x_train  # 과적합 상황 보여주기
# y_test = y_train  # train 데이터로 best_estimator_로 예측 후 점수를 내면
                    # best_score_ 나온다.

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
####################################################################
# print(model.cv_results_)
# {'mean_fit_time': array([0.00059786, 0.0003993 , 0.00039377, 0.00039816, 0.00039358,... >> 평균 훈련 시간  : 42번

aaa = pd.DataFrame(model.cv_results_)
# print(bbb)
bbb = aaa[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score']]
    #  'split0_test_score','split1_test_score', 'split2_test_score',
    #  'split3_test_score','split4_test_score']]

print(bbb)

    mean_fit_time  std_fit_time  mean_score_time  std_score_time  ... split4_test_score mean_test_score std_test_score rank_test_score
0        0.000194      0.000388         0.000411        0.000504  ...          0.958333        0.958333       0.045644               6       
1        0.000394      0.000483         0.000399        0.000488  ...          0.958333        0.958333       0.045644               6       
2        0.000195      0.000389         0.000204        0.000409  ...          0.958333        0.958333       0.045644               6    
'''



'''
최적의 매개 변수 : SVC(C=1, kernel='linear')
최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
best_score_ : 0.9666666666666666 >>> train에서 가장 좋은 값 //
model.score : 1.0                >>> test(or predict)에서 가장 좋은 값
accuracy_score : 1.0

#############################

x_test = x_train
y_test = y_train
best_score_ : 0.9833333333333332
model.score : 0.975
accuracy_score : 0.975

model.best_estimator_ = model.score(x_test, y_test) : 동일하게 생각해도 된다. 
'''
