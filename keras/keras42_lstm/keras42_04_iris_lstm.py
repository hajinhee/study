from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_iris
import numpy as np
from pandas import get_dummies
from icecream import ic
import pandas as pd

#1. 데이터로드 및 정제
datasets = load_iris()
x = datasets.data  # (150, 4)         
y = datasets.target  # (150, )
ic(np.unique(y, return_counts=True))  # array([0, 1, 2]), array([50, 50, 50] -> 다중분류 -> 원핫인코딩 

# 컬럼명 추가하기 위해 pandas로 변환
x = pd.DataFrame(x, columns=datasets.feature_names) 

# 상관관계 알기 위해 x에 'ydata'라는 컬럼 만들어 y값 담기
x['ydata'] = y  
ic(x.corrwith(x['ydata']))
'''
sepal length (cm)    0.782561
sepal width (cm)    -0.426658
petal length (cm)    0.949035
petal width (cm)     0.956547
ydata                1.000000
'''

# 불필요한 컬럼 제거
x.drop(['ydata'], axis=1, inplace=True)  
ic(x.shape)  # (150, 4)  

# 원핫인코딩
y = get_dummies(y)  # (150, 3)

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# 스케일링
scaler = StandardScaler()  # 분류모델에 유용
# scaler = MinMaxScaler()  # 회귀모델에 유용
# scaler = RobustScaler()   
# scaler = MaxAbsScaler()   

# RNN 사용시 
x_train = scaler.fit_transform(x_train).reshape(len(x_train), 2, 2)  # (120, 2, 2)
x_test = scaler.transform(x_test).reshape(len(x_test), 2, 2)
# DNN 사용시
# x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1))  --> 행은 데이터 개수만큼, 열은 컬럼 개수만큼 reshape 된다.
# x_test = scaler.transform(x_test.reshape(len(x_test), -1))

ic(x_train.shape)

#2. 모델링
model = Sequential()
model.add(SimpleRNN(10, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))  # SimpleRNN    
model.add(LSTM(10, return_sequences=True, activation='relu'))  # LSTM
model.add(GRU(10, return_sequences=False, activation='relu'))  # GRU
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(4))
model.add(Dense(3, activation='softmax'))  # 'softmax' -> 다중분류

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
es = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=1, callbacks=[es])      

#4. 평가, 예측   
loss = model.evaluate(x_test, y_test)

print('[loss]: ', round(loss[0], 4))
print('[accuracy]: ', round(loss[1], 4))

'''
[loss]:  0.6851
[accuracy]:  0.9
'''