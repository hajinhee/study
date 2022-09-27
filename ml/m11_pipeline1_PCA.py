import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from catboost import CatBoostClassifier

#1. 데이터

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 100)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test  = scaler.fit_transform(x_test)

from sklearn.decomposition import PCA  # 컬럼을 압축하다. 

# PCA : 비지도 학습과 비슷하다.
 
#2. 모델
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# model = SVC()
# model = make_pipeline(MinMaxScaler(),StandardScaler(), SVC())  
# #fit transform이 가능한 것들은 계속해서 엮을 수 있다.
model = make_pipeline(MinMaxScaler(),PCA(), SVC())


#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('model.score :', result)
print('accuracy_score :', acc)