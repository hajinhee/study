from pandas.core.frame import DataFrame
from scipy.sparse import data
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pandas import get_dummies,DataFrame
import inspect, os
import math
from icecream import ic

#1 데이터
datasets = fetch_covtype()
x = datasets.data  # (581012, 54)         
y = datasets.target  # (581012,)       
ic(np.unique(y, return_counts=True))  # (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]) --> 다중분류 

y = get_dummies(y)  # 원핫인코딩

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49)
ic(x_train.shape)  # (522910, 54)
ic(x_test.shape)  # (58102, 54)

scaler = MinMaxScaler()      
x_train = scaler.fit_transform(x_train).reshape(len(x_train), 9, 6, 1)  # (522910, 9, 6, 1) 
x_test = scaler.transform(x_test).reshape(len(x_test), 9, 6, 1)  # (58102, 9, 6, 1)


#2.모델링
model = Sequential()
model.add(Conv2D(4, kernel_size=(2, 1), strides=1, padding='valid', input_shape=(9, 6, 1), activation='relu'))
model.add(MaxPooling2D(2, 2))                                                                             
model.add(Conv2D(4, kernel_size=(2, 1), strides=1, padding='valid', activation='relu'))                                                                                       # 1,1,10
model.add(Conv2D(4, kernel_size=(2, 2), strides=1, padding='valid', activation='relu'))               
model.add(MaxPooling2D(2, 2))                                                                                             
model.add(Flatten())       
model.add(Dense(40))
model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, baseline=None, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras35_2_diabetes{krtime}.hdf5')
model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.111111, callbacks=[es])

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('[loss] : ', loss[0])
print('[accuracy] : ', loss[1])

acc= str(round(loss[1], 4))
model.save(f"./_save/keras35_6_fetch_acc_Min_{acc}.h5")


'''
[loss] :   0.7988 
[accuracy] :   0.6323
'''
