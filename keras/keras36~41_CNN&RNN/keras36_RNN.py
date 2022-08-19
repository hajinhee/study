import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from icecream import ic

#1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])  # (4, 3) 2d
y = np.array([4,5,6,7])  # (4, )                           
x = x.reshape(4, 3, 1)  # (4, 3, 1) 3d

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(32, activation='linear', input_shape=(3, 1)))  # input에는 모든 행을 제외
model.add(Dense(10))        
model.add(Dense(8))                 
model.add(Dense(4))                 
model.add(Dense(2))                 
model.add(Dense(1))                         

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x, y, epochs=100, batch_size=1, callbacks=[es])  

#4. 평가, 예측
model.evaluate(x, y)
y = np.array([5, 6, 7])  # (3, ) 1d
'''
[5, 6, 7]
'''
y_pred = y.reshape(1, 3, 1)  # (1, 3, 1) 3d
'''
[[[5],
  [6],
  [7]]]
'''
result = model.predict(y_pred)  # input_shape를 원래 입력값과 똑같이 맞춰줘야 한다. 
ic(result)
'''
[[8.040146]]
'''


