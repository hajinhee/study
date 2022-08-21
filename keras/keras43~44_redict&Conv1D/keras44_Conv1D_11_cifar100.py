from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout ,SimpleRNN, LSTM, GRU, Activation, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
from tensorflow.keras.datasets import cifar100
import numpy as np
from pandas import get_dummies

#1. 데이터 로드 및 정제
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape)  # (50000, 32, 32, 3) 4d
print(y_train.shape)  # (50000, 1)
print(x_test.shape)  # (10000, 32, 32, 3) 4d
print(y_test.shape)  # (10000, 1)
print(np.unique(y_train, return_counts=True))  # 0 ~ 99 값 --> 다중분류      

# 원핫인코딩(1차원 데이터만 가능) 
y_train = get_dummies(y_train.reshape(len(y_train)))  # (50000, 1) -> (50000, ) -> (50000, 100)
y_test = get_dummies(y_test.reshape(len(y_test)))  # (10000, 1) -> (10000, ) -> (10000, 100)

# 스케일링(2차원 데이터만 가능)
scaler = StandardScaler()   
# scaler = MinMaxScaler()   
# scaler = RobustScaler()   
# scaler = MaxAbsScaler()   

# RNN 사용 시 (4차원 -> 2차원 -> 3차원)
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(len(x_train), 384, 8)
x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(len(x_test), 384, 8)                               

# DNN 사용 시 (4차원 -> 2차원)
# x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1))
# x_test = scaler.transform(x_test.reshape(len(x_test), -1))
# print(x_train.shape)

#2. 모델링
model = Sequential()
model.add(SimpleRNN(10, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(LSTM(10, return_sequences=True, activation='relu'))    
model.add(GRU(10, return_sequences=False, activation='relu'))    
# model.add(Dense(50,input_dim= x_train.shape[1]))                                               
# model.add(Conv1D(10, 2, input_shape=(x_train.shape[1], x_train.shape[2])))  
# model.add(Flatten()) 
model.add(Dense(64))  
model.add(Dense(32))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu')) 
model.add(Dense(4))
model.add(Dense(100, activation='softmax'))  

#3. 컴파일, 훈련
start = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
# es = EarlyStopping(monitor='val_loss', patience=1, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=20, batch_size=10000, validation_split=0.2, verbose=1)    
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측 
loss = model.evaluate(x_test, y_test)
print('[loss]: ', round(loss[0], 4))
print('[accuracy]: ', round(loss[1], 4))


'''
******RNN*******
[loss]:  4.5559
[accuracy]:  0.0177

*******Comv1D*******
[loss]:  4.4839
[accuracy]:  0.029
'''