# n_component > 0.95 이상
# xgb, gridSearch 또는 randomsearch를 사용할 것
# m16결과를 뛰어넘어랏

import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, r2_score
import time
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x = np.append(x_train, x_test, axis=0) 
# x_train = x_train.reshape(60000, -1)  # (60000, 784)
# x_test = x_test.reshape(10000, -1) 
# x= x.reshape(70000, 784)
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# pca
pca = PCA(n_components=154)
# x = pca.fit_transform(x)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

parameters = [
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01],
    'max_depth':[4,5,6]}]
    # {'n_estimators':[90, 100, 110], 'learning_rate':[0.1, 0.001, 0.01],
    # 'max_depth':[4,5,6], 'colsample_bytree' :[0.6, 0.9, 1]},
    # {'n_estimators':[90, 100, 110], 'learning_rate':[0.1, 0.001, 0.01],
    # 'max_depth':[4,5,6], 'colsample_bytree' :[0.6, 0.9, 1],
    # 'colsample_bylevel':[0.6,0.7,0.9]}]
# parameters = {"XGB__n_estimators":[90,100,110,200,300], "XGB__learning_rate":[0.1,0.3,0.001,0.01],
#               "XGB__max_depth":[4,5,6],'XGB__use_label_encoder':[False],
#             "XGB__colsample_bytree":[0.6,0.9,1],"XGB__colsample_bylevel":[0.6,0.7,0.9],
#             "XGB__random_state":[66],"XGB__eval_metric":['error']}

#2. 모델
from sklearn.pipeline import make_pipeline, Pipeline
# pipe = Pipeline([('mm',MinMaxScaler()),('xg', XGBClassifier(eval_metric='merror'))])
# model = GridSearchCV(pipe, parameters, cv = 2, verbose = 3, n_jobs=-1)
model = RandomizedSearchCV(XGBClassifier(use_label_encoder=False), parameters, cv=3, verbose=3,  
                     refit=True, n_jobs=-1)

#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train,eval_metric='merror')
end = time.time()

#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print('model.score :', result)
print('accuracy_score :', r2)
print('걸린 시간 :', end - start)
