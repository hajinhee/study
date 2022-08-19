from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import numpy as np
from icecream import ic
import pandas as pd

#1. 데이터 로드 및 정제 
datasets = load_boston()
x = datasets.data  # (506, 13) 2d        
y = datasets.target 
# ic(np.unique(y, return_counts=True))  # 값이 수도 없이 많음 --> 회귀모델    

# 컬럼 추가를 위해 numpy -> pandas 변환 
x = pd.DataFrame(x, columns=datasets.feature_names)  # 판다스로 변환 후 판다스 기능으로 데이터셋 feature_names를 columns으로 추가
'''
type(x): <class 'pandas.core.frame.DataFrame'>
x.shape: (506, 13)
x.ndim: 2
'''
x = x.to_numpy()  # 컬럼 추가 후 다시 pandas -> numpy 변환
'''
type(x): <class 'numpy.ndarray'>
x.shape: (506, 13)
x.ndim: 2
'''
x = x.reshape(len(x), 13, 1)  # RNN에 넣기 위해 3차원 데이터로 변환 
'''
x.shape: (506, 13, 1)
x.ndim: 3
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)

# 스케일링에는 2차원 데이터만 입력 가능하다.
scaler = MinMaxScaler()
# scaler = StandardScaler()   
# scaler = RobustScaler()   
# # scaler = MaxAbsScaler()  

# RNN 사용 시 (스케일링(2차원) -> RNN(3차원))
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)  # (404, 13) 2d -> (404, 13, 1) 3d
x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(x_test.shape)

# DNN 사용 시 (스케일링(2차원) -> DNN(2차원))
# x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1))  # (404, 13) 2d
# x_test = scaler.transform(x_test.reshape(len(x_test), -1))

#2. 모델링  
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))  # SimpleRNN
model.add(LSTM(10, return_sequences=False, activation='relu'))  # LSTM                       
# model.add(GRU(10, return_sequences=False, activation='relu'))  # GRU
model.add(Dense(50))                                                             
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(15, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(5))
model.add(Dense(1))  # default는 'linear' -> 회귀모델

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')  # 'mse' -> 회귀모델
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1, callbacks=[es])      

#4. 평가, 예측 --> 회귀모델은 [r2] 분류모델은 [accuracy]
loss = model.evaluate(x_test, y_test)

ic("----------------------loss값-------------------------")
ic(round(loss, 4))
y_predict = model.predict(x_test)

ic("=====================r2score=========================")
r2 = r2_score(y_predict, y_test) 
ic(round(r2, 4))

'''
'----------------------loss값-------------------------'
39.573
'=====================r2score========================='
0.2514
'''
