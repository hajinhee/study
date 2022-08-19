import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6], [4,5,6,7]])  # (4, 4) 2d
y = np.array([5,6,7,8])  # (4, ) 1d                          
x = x.reshape(4, 1, 4)  # (4, 1, 4) 3d

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(5, input_shape=(1, 4)))  # SimpleRNN
model.add(LSTM(5, input_length=1, input_dim=4))  # LSTM
model.add(Dense(9))        
model.add(Dense(8))                 
model.add(Dense(4))                 
model.add(Dense(2))                 
model.add(Dense(1))                         
# model.summary()

'''
[SimpleRNN]
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 simple_rnn (SimpleRNN)      (None, 5)                 50

 dense (Dense)               (None, 9)                 54

 dense_1 (Dense)             (None, 8)                 80

 dense_2 (Dense)             (None, 4)                 36

 dense_3 (Dense)             (None, 2)                 10

 dense_4 (Dense)             (None, 1)                 3

=================================================================
Total params: 233
Trainable params: 233
Non-trainable params: 0
_________________________________________________________________

[LSTM]
_________________________________________________________________
Layer (type)                 Output Shape              Param #      
=================================================================
lstm (LSTM)                  (None, 5)                 200          *** SimpleRnn과 4배 차이
_________________________________________________________________       LSTM은 RNN의 문제를 셀상태(Cell state)와 여러 개의 게이트(gate)를 가진 셀이라는 유닛을 통해 해결한다.
dense (Dense)                (None, 9)                 54               이 유닛은 시퀀스상 멀리있는 요소를 잘 기억할 수 있도록 한다.
_________________________________________________________________       셀 상태는 기존 신경망의 은닉층이라고 생각할 수 있다.
dense_1 (Dense)              (None, 8)                 80               셀상태를 갱신하기 위해 기본적으로 3가지의 게이트가 필요하다.
_________________________________________________________________       3개의 게이트(Forget, input, output)와 1개의 cell state가 있어서 4배가 된다.
dense_2 (Dense)              (None, 4)                 36
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 10
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 3
=================================================================
Total params: 383
Trainable params: 383
Non-trainable params: 0
_________________________________________________________________
'''

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=500, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x, y, epochs=100, batch_size=1, callbacks=[es])  

#4. 평가, 예측
model.evaluate(x, y)
y_pred = np.array([4,5,6,7]).reshape(1, 1, 4)  # (1, 1, 4) 3d
result = model.predict(y_pred)  
print(result)

'''
SimpleRNN: [[7.730986]]
LSTM: [[8.013119]]      --> 정확도가 더 높다.
'''