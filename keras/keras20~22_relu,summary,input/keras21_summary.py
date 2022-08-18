from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
import numpy as np

#1. 데이터
x =  np.array([1,2,3])  # (3, )
y =  np.array([1,2,3])  # (3, )

#2. 모델구성
model = Sequential()
model.add(Dense(8, input_dim=1)) 
model.add(Dense(5, activation="relu")) 
model.add(Dense(8, activation="sigmoid"))
model.add(Dense(4))    
model.add(Dense(1))  # 회귀모델은 출력층에 활성화 함수를 넣지 않는다.

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #      바이어스 값이 있어서 1씩 더해서 다음값이랑 곱한다
=================================================================
dense (Dense)                (None, 5)                 10           (1+1)*5     10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18           (5+1)*3     18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16           (3+1)*4     16
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 10           (4+1)*2     10
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 3            (2+1)*1      3
=================================================================
Total params: 57
Trainable params: 57
Non-trainable params: 0
_________________________________________________________________
'''


#3. 컴파일, 훈련     
model.compile(loss='mse', optimizer='adam') 
model.fit(x, y, epochs=2000, batch_size=1) 

#4. 평가, 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)
result = model.predict([4]) 
print('4의 예측값 : ', result)
