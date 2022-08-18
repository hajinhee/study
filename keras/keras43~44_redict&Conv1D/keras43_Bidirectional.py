# RNN방식을 한단계 더 보완해주기위해 양방향으로 순환시켜서 더 좋은 성능 향상을 기대해본다.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])     

y = np.array([4,5,6,7])                             

#print(x.shape, y.shape)     # (4,3) (4,)

x = x.reshape(4,3,1)    

#2. 모델구성
model = Sequential()
#model.add(SimpleRNN(20,input_shape=(3,1), return_sequences=True)) 
model.add(Bidirectional(SimpleRNN(20), input_shape=(3,1)))   # layer의 instance값을 적어야한다. 어떤 RNN을 쓸것인지 명시해줘야한다. 원래의 RNN에
model.add(Dense(10))                                         # (Bidirectional)더 감싸주고 뒤에 input_shape 값을 넣어준다. 
model.add(Dense(8))                 
model.add(Dense(4))                 
model.add(Dense(2))                 
model.add(Dense(1))                         
model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
simple_rnn (SimpleRNN)       (None, 3, 10)             120
_________________________________________________________________
bidirectional (Bidirectional (None, 20)                420
_________________________________________________________________
dense (Dense)                (None, 10)                210
_________________________________________________________________
dense_1 (Dense)              (None, 8)                 88
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 36
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 10
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 3
=================================================================
Total params: 887
Trainable params: 887
Non-trainable params: 0
_________________________________________________________________
'''
'''
#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam') #mae도있다.
es = EarlyStopping(monitor="loss", patience=500, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x,y, epochs=10000, batch_size=1, callbacks=[es])  

#4. 평가, 예측

model.evaluate(x,y)
y_pred = np.array([5,6,7]).reshape(1,3,1)
result = model.predict(y_pred)   
print(result)
'''
