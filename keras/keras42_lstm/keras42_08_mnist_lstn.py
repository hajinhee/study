from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout ,SimpleRNN, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
import numpy as np
from pandas import get_dummies
import pandas as pd

#1.데이터로드 및 정제
datasets = mnist.load_data()
(x_train, y_train), (x_test, y_test) = datasets
print(x_train.shape)  # (60000, 28, 28) 3d
print(x_test.shape)   # (10000, 28, 28) 3d
print(y_train.shape)  # (60000,) 1d
print(y_test.shape)   # (10000,) 1d

print(np.unique(y_train, return_counts=True))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949] --> 다중분류

# 분류모델 -> 원핫인코딩
y_train = get_dummies(y_train)  # (60000, 10) 2d
y_test = get_dummies(y_test)  # (10000, 10) 2d

# 스케일링
scaler = StandardScaler() 
# scaler = MinMaxScaler()   
# scaler = RobustScaler()   
# scaler = MaxAbsScaler()   

# RNN사용시 
# 자동으로 3차원데이터를 2차원으로 만들어서 스케일링 적용하고 다시 3차원으로 적용해줌.
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)  # (60000, 28, 28) -> (60000, 784) -> (60000, 28, 28)
x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(x_test.shape)  # (10000, 28, 28) -> (10000, 784) -> (10000, 28, 28)

# DNN 사용 시
# x_train = x_train.reshape(len(x_train), 784)  # (60000, 784) 
# x_test = x_test.reshape(len(x_test), 784)  # (10000, 784)

#2. 모델링 
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))  # SimpleRNN    
model.add(LSTM(10, return_sequences=True, activation='relu'))  # LSTM                               
model.add(GRU(10, return_sequences=False, activation='relu'))  # GRU                        
# model.add(Dense(50, input_dim= x.shape[1]))                                                               
model.add(Dense(64))                                                                       
model.add(Dense(32))
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(4))
model.add(Dense(10, activation='softmax')) 

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_accuracy', patience=50, mode='max', verbose=1, baseline=None, restore_best_weights=True)  
model.fit(x_train, y_train, epochs=100, batch_size=6000, validation_split=0.2, verbose=1, callbacks=[es])       

#4. 평가, 예측      
loss = model.evaluate(x_test, y_test)
print('[loss]: ', round(loss[0], 4))
print('[accuracy]: ', round(loss[1], 4))

'''
[loss]:  0.3648
[accuracy]:  0.8912
'''