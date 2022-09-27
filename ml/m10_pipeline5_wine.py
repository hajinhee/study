import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

#1. 데이터

datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 100)

#2. 모델
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

model = make_pipeline(MinMaxScaler(), CatBoostClassifier())

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('model.score :', result)
print('accuracy_score :', acc)
print('걸린 시간 :', end-start)

'''
model.score : 1.0
accuracy_score : 1.0
걸린 시간 : 2.0078697204589844
'''