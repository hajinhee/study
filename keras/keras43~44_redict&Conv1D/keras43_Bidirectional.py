import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

'''
RNN방식을 한 단계 더 보완하기 위해 양방향으로 순환시켜서 더 좋은 성능 향상을 기대해본다.
'''

#1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])  # (4, 3)
y = np.array([4, 5, 6, 7])  # (4, )                           
x = x.reshape(4, 3, 1)  # (4, 3, 1) 

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(20, input_shape=(3, 1), return_sequences=True))  # 단방향 SimpleRNN
model.add(Bidirectional(SimpleRNN(20), input_shape=(3, 1)))  # 양방향 SimpleRNN
model.add(Dense(10))                                        
model.add(Dense(8))                 
model.add(Dense(4))                 
model.add(Dense(2))                 
model.add(Dense(1))                         

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x, y, epochs=300, batch_size=1, callbacks=[es])  

#4. 평가, 예측
model.evaluate(x, y)
y_pred = np.array([5, 6, 7]).reshape(1, 3, 1)  
result = model.predict(y_pred)   
print('[y_pred]: ', result)  #  [[7.818365]]  loss: 4.3026e-05

