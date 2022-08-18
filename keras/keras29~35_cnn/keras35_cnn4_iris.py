from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import inspect, os
from icecream import ic

#1 데이터
datasets = load_iris()
x = datasets.data  # (150, 4)          
y = datasets.target  # (150,)       
print(np.unique(y, return_counts=True))  # (array([0, 1, 2]), array([50, 50, 50], dtype=int64)) --> 다중분류

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49)

scaler = MinMaxScaler()  
x_train = scaler.fit_transform(x_train).reshape(len(x_train), 2,2,1)  # (135, 2, 2, 1)  
x_test = scaler.transform(x_test).reshape(len(x_test), 2,2,1)  # (135, 2, 2, 1)  

y_train = to_categorical(y_train)  # (135, 3) 
y_test = to_categorical(y_test)  # (135, 3) 

#2.모델링
model = Sequential()
model.add(Conv2D(4, kernel_size=(2, 2), strides=1, padding='same', input_shape=(2, 2, 1), activation='relu'))                                                                      # 1,1,10
model.add(MaxPooling2D(2, 2))                                                                                                 
model.add(Flatten())       
model.add(Dense(60))
model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, baseline=None, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras35_2_diabetes{krtime}.hdf5')
model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.111111, callbacks=[es])

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('[loss] : ', loss[0])
print('[accuracy] : ', loss[1])

acc= str(round(loss[1], 4))
model.save(f"./_save/keras35_4_iris_acc_Min_{acc}.h5")


'''
[loss] :  0.013341540470719337
[accuracy] :  1.0
'''
