import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from icecream import ic
#1. 데이터
x = np.array([[1,2,3],                              
             [2,3,4],                               
             [3,4,5],
             [4,5,6]])  # (4, 3) 2d
y = np.array([4,5,6,7])  # (4, ) 1d
x = x.reshape(4, 3, 1)  # (4, 3, 1) 3d

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(3, 1)))                    
model.add(SimpleRNN(10, input_length=3, input_dim=1))   
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()
'''
RNN 파라미터 개수 구하는 방법
( unit 개수 * unit 개수 ) + ( input_dim(feature) 수 * unit 개수 ) + ( 1 * unit 개수)
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 10)                120       

 dense (Dense)               (None, 10)                110       

 dense_1 (Dense)             (None, 1)                 11        

=================================================================
Total params: 241
Trainable params: 241
Non-trainable params: 0
_________________________________________________________________

'''

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')   
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
model.evaluate(x, y)
x2 = np.array([5,6,7]).reshape(1,3,1)  # (1, 3, 1)
result = model.predict(x2)
ic(result)  # [[0.39999834]]
