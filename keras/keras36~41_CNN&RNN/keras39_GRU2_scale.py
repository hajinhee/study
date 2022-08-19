import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
import time
from icecream import ic

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9],
              [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])  # (13, 3)   
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])  # (13, )                         
x = x.reshape(13, 3, 1)  # (13, 3, 1)

#2. 모델구성
model = Sequential()
model.add(GRU(5, activation='relu', input_shape=(3,1)))  # GRU
model.add(Dense(80))
model.add(Dense(60, activation='relu'))
model.add(Dense(40))
model.add(Dense(20, activation='relu'))        
model.add(Dense(10))                 
model.add(Dense(5))                 
model.add(Dense(1))    
                     
#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=500, mode='min', verbose=2, baseline=None, restore_best_weights=True)

start = time.time()
model.fit(x, y, epochs=1000, batch_size=1, callbacks=[es])  
end = time.time()-start
ic("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
model.evaluate(x, y)
y_pred = np.array([50 ,60, 70]).reshape(1, 3, 1)
result = model.predict(y_pred)  
ic(result)

'''
epochs=1000, batch_size=1  [loss]: 0.0081  [result]: [[80.32456]]
'''