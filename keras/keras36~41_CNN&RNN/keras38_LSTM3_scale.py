import imp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from icecream import ic
from tensorflow.keras.optimizers import Adam

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], 
              [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])  # (13, 3) 2d
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])  # (13, ) 1d                        
x = x.reshape(13, 3, 1)  # (13, 3, 1) 3d

#2. 모델구성
model = Sequential()
model.add(LSTM(5, activation='relu', input_shape=(3, 1)))  # LSTM
model.add(Dense(80))
model.add(Dense(60, activation='relu'))
model.add(Dense(40))
model.add(Dense(20, activation='relu'))        
model.add(Dense(10))                 
model.add(Dense(5))                 
model.add(Dense(1))          
# model.summary()

#3. 컴파일,훈련
es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=1, verbose=1, callbacks=[es])  

#4. 평가, 예측
model.evaluate(x, y)
y_pred = np.array([50,60,70]).reshape(1, 3, 1)  # (1, 3, 1) 3d
result = model.predict(y_pred)  
ic(result)  # [[81.02125]]
