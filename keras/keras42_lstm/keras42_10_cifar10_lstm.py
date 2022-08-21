from binascii import a2b_base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout ,SimpleRNN, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
import numpy as np
from pandas import get_dummies

#1. 데이터 로드 및 정제
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)  # (50000, 32, 32, 3) 3d 
print(x_test.shape)  # (10000, 32, 32, 3) 3d
print(y_train.shape)  # (50000, 1) 2d 
print(y_test.shape)  # (10000, 1) 2d
print(np.unique(y_train, return_counts=True))  # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])) --> 다중분류

# 원핫인코딩(1차원 데이터만 가능)
y_train = y_train.reshape(len(y_train))  # (50000, ) 1d --> 원핫인코딩은 1차원 형태의 데이터를 처리하므로 1차원 벡터형식으로 reshape 한다.
y_train = get_dummies(y_train)  # (50000, 10) 2d
y_test = y_test.reshape(len(y_test))  # (10000, ) 1d 
y_test = get_dummies(y_test)  # (10000, 10) 2d 

# 스케일링(2차원 데이터만 가능)
# scaler = StandardScaler()   
scaler = MinMaxScaler()   
# scaler = RobustScaler()   
# scaler = MaxAbsScaler()   

# RNN 사용 시 (4차원 -> 2차원 -> 3차원)
# x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(len(x_train), 384, 8)
# x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(len(x_test), 384, 8)                               

# DNN 사용 시 (4차원 -> 2차원)
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1))
x_test = scaler.transform(x_test.reshape(len(x_test), -1))
print(x_train.shape)

#2. 모델링
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))   
# model.add(LSTM(10, return_sequences=True, activation='relu')) 
# model.add(GRU(10, return_sequences=False, activation='relu')) 
model.add(Dense(50, input_dim=x_train.shape[1]))  
model.add(Dropout(0.5)) 
model.add(Dense(64))
model.add(Dropout(0.5))     
model.add(Dense(32))
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(4))
model.add(Dense(10, activation='softmax'))    

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])    
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=5000, validation_split=0.2, verbose=1, callbacks=[es])  

#4. 평가, 예측   
loss = model.evaluate(x_test, y_test)
print('[loss]: ', round(loss[0], 4))
print('[accuracy]: ', round(loss[1], 4))

'''
[loss]:  1.7494
[accuracy]:  0.3612
'''