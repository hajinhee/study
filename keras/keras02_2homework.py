# import tensorflow as tf
from tensorflow.keras.models import Sequential # 시퀀셜 모델과
from tensorflow.keras.layers import Dense # 덴스 레이어를 쓸수있다. 
import numpy as np

x =  np.array([1,2,3]) # shape=(3, 1)
y =  np.array([1,2,3])

model = Sequential()
model.add(Dense(10, input_dim=1)) # input layer
model.add(Dense(40)) 
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20)) 
model.add(Dense(10)) # 중간에 있는 레이어는 hidden layer 
model.add(Dense(1)) # output layer

#3. 컴파일
model.compile(loss='mse', optimizer='adam') 

#4. 훈련
model.fit(x, y, epochs=50, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)
result = model.predict([4]) 
print('4의 예측값 : ', result)

