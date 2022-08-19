import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
'''
[GRU, Gated Recurrent Unit]
LSTM에서는 출력, 입력, 삭제 게이트라는 3개의 게이트가 존재한 반면,
GRU에서는 [업데이트 게이트]와 [리셋 게이트] 두 가지 게이트만이 존재한다.
GRU는 LSTM보다 학습 속도가 빠르다고 알려져있지만 여러 평가에서 GRU는 LSTM과 비슷한 성능을 보인다.
'''

#1. 데이터
x = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6], [4,5,6,7]])  # (4, 4) 2d    
y = np.array([5,6,7,8])  # (4, ) 1d                          
x = x.reshape(4, 2, 2)  # (4, 2, 2) 3d

#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(6, input_shape=(2, 2)))  # SimpleRNN        
# model.add(LSTM(6, input_shape=(2, 2)))  # LSTM        
model.add(GRU(6, input_shape=(2, 2)))  # GRU        
model.add(Dense(5))               
model.add(Dense(2))                 
model.add(Dense(1))                         
model.summary()

'''
[SimpleRNN]
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 6)                 54

 dense (Dense)               (None, 5)                 35

 dense_1 (Dense)             (None, 2)                 12

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 104
Trainable params: 104
Non-trainable params: 0
_________________________________________________________________

[LSTM]
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 6)                 216

 dense (Dense)               (None, 5)                 35

 dense_1 (Dense)             (None, 2)                 12

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 266
Trainable params: 266
Non-trainable params: 0
_________________________________________________________________

[GRU]
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 gru (GRU)                   (None, 6)                 180       

 dense (Dense)               (None, 5)                 35        

 dense_1 (Dense)             (None, 2)                 12

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 230
Trainable params: 230
Non-trainable params: 0
_________________________________________________________________
'''

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor='loss', patience=500, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x,y, epochs=2000, batch_size=1, callbacks=[es])  

#4. 평가, 예측
model.evaluate(x, y)
y_pred = np.array([5,6,7,8]).reshape(1, 2, 2)  # (1, 2, 2) 3d
result = model.predict(y_pred)  
print(result)  

'''
epoch=100 -> [loss]: 0.0866  [result]: [[7.7727413]]
epoch=200 -> [loss]: 0.0532  [result]: [[7.965357]]
epoch=250 -> [loss]: 0.0221  [result]: [[8.226958]]
epoch=500 -> [loss]: 0.0050  [result]: [[8.542177]]
epoch=1000 -> [loss]: 1.1764e-04  [result]: [[8.781873]]
epoch=2000 -> [loss]: 1.0519e-05  [result]: [[8.85312]]
'''

