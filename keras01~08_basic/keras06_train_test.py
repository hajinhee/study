from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x_train = np.array([1,2,3,4,5,6,7]) # shape=(7, )  
x_test = np.array([8,9,10]) # shape=(3, )        
y_train = np.array([1,2,3,4,5,6,7]) # shape=(7, ) 
y_test = np.array([8,9,10]) # shape=(3, ) 

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=400, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) # 평가
print('loss: ', loss) 
result = model.predict([11])
print('[11]의 예측값 : ', result)

# epochs=100, batch=1 [11]의 예측값 :  [[10.990736]]
# epochs=200, batch=1 [11]의 예측값 :  [[10.999658]]
# epochs=300, batch=1 [11]의 예측값 :  [[10.999997]]
# epochs=400, batch=1 [11]의 예측값 :  [[11.]]