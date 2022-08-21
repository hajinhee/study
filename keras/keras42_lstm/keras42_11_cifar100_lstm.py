from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout ,SimpleRNN, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar100
import numpy as np
from pandas import get_dummies

#1. 데이터 로드 및 정제
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape)  # (50000, 32, 32, 3) 4d,  (50000, 1) 2d
print(x_test.shape, y_test.shape)  # (10000, 32, 32, 3) 4d,  (10000, 1) 2d
print(np.unique(y_train, return_counts=True))  # 0 ~ 99 다중분류    
'''
[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99], 
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
500, 500, 500, 500, 500, 500, 500, 500, 500]
'''

# 다중분류 -> 원핫인코딩(1차원 데이터만 입력 가능)
y_train = y_train.reshape(len(y_train))  # (50000, ) 1d
y_train = get_dummies(y_train)  # (50000, 100) 2d
y_test = y_test.reshape(len(y_test))  # (10000, ) 1d
y_test = get_dummies(y_test)  # (10000, 100) 2d

# 스케일링
# scaler = StandardScaler()   
scaler = MinMaxScaler()   
# scaler = RobustScaler()   
# scaler = MaxAbsScaler()   

# RNN 사용 시 
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(len(x_train), 384, 8)  # (50000, 32, 32, 3) -> (50000, 3072) -> (50000, 384, 8)
x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(len(x_test), 384, 8)  # (10000, 32, 32, 3) -> (10000, 3072) -> (10000, 384, 8)

# DNN 사용 시
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2.모델링
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))  
model.add(LSTM(10, return_sequences=True, activation='relu'))                             
model.add(GRU(10, return_sequences=False, activation='relu'))    
# model.add(Dense(50, input_dim= x_train.shape[1]))                                              
model.add(Dense(64))                                                   
model.add(Dense(32))
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(4))
model.add(Dense(100, activation='softmax'))  

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=10000, validation_split=0.2, verbose=1, callbacks=[es])   

#4. 평가, 예측     
loss = model.evaluate(x_test, y_test)
print('[loss]: ', round(loss[0], 4))
print('[accuracy]: ', round(loss[1], 4))

'''
[loss]:  4.3321
[accuracy]:  0.0391
'''