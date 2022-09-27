import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd

#1. 데이터 
path = '../_data/kaggle/bike/'  
train = pd.read_csv(path+'train.csv')  
# print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
# print(test.shape)    # (6493, 9)
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 

y = train['count']

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 100)

#2. 모델
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

model = make_pipeline(MinMaxScaler(), CatBoostRegressor())

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)

r2 = r2_score(y_predict, y_test)
print(r2)
print('걸린 시간 :', end-start)