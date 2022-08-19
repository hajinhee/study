from enum import unique
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import numpy as np
from icecream import ic
import pandas as pd

#1. 데이터로드 및 정제
datasets = load_diabetes()
x = datasets.data  # (442, 10) 
y = datasets.target  # (442, ) 수많은 label -> 회귀모델

'''
[unique value 추출 방법]

# 방법 1. numpy의 np.unique(return_counts=True) ---> ndarray만 가능
# ic(np.unique(y, return_counts=True))       

# 방법 2. pandas의 value_counts() ---> Dataframe column 혹은 Series만 가능
y = pd.DataFrame(y)  # numpy --> pandas 변환
ic(y.value_counts())  
'''

# 컬럼명 추가하기 위해 pandas로 변환
x = pd.DataFrame(x, columns=datasets.feature_names)

# 상관관계 분석 위해 y(정답) 값 x에 'ydata' 컬럼으로 추가
x['ydata'] = y
ic(x.corrwith(x['ydata']))
'''
age      0.187889
sex      0.043062
bmi      0.586450
bp       0.441482
s1       0.212022
s2       0.174054
s3      -0.394789
s4       0.430453
s5       0.565883
s6       0.382483
ydata    1.000000
'''

# 불필요한 컬럼 삭제
x.drop(['sex', 'ydata'], axis=1, inplace=True)  

# 이후 작업 위해 다시 numpy로 변환 -> 3차원으로 변환
x = x.to_numpy().reshape(len(x), 3, 3)    # (442, 9) -> (442, 3, 3)

# 데이터셋 분리
x_train, x_test, y_train,  y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)

# 스케일링 
scaler = MinMaxScaler()  
# RNN 사용 시 (3차원 -> 2차원 -> 3차원)
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(x_test.shape)
# DNN 사용 시 (3차원 -> 2차원)
# x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1))
# x_test = scaler.transform(x_test.reshape(len(x_test), -1))

#2. 모델링  
model = Sequential()
model.add(SimpleRNN(10, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))       
model.add(LSTM(10, return_sequences=True, activation='relu'))                         
model.add(GRU(10, return_sequences=False, activation='relu')) 
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(4))
model.add(Dense(1))    

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')   
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1, callbacks=[es])      

#4. 평가, 예측      
loss = model.evaluate(x_test, y_test)
ic('loss: ', round(loss, 4))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
ic('r2_score: ', round(r2, 4))

'''
[loss]:  2060.7495
[r2_score]:  0.6132
'''