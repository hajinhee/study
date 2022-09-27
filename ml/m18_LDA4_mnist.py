import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,784) 
x_test = x_test.reshape(10000, 784)
print('LDA 전:',x_train.shape)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


lda = LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

print('LDA 후:',x_train.shape)

#2. 모델
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from ngboost import NGBClassifier
model = LGBMClassifier()

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print('결과:', results)
print('걸린 시간', end-start)

'''
- XGBOOST
LDA 전: (60000, 784)
LDA 후: (60000, 9)
결과: 0.9163
걸린 시간 34.25585889816284

- LGBM
LDA 전: (60000, 784)
LDA 후: (60000, 9)
결과: 0.9168
걸린 시간 1.876582384109497

- catboost
결과: 0.9186
걸린 시간 33.12629771232605
'''