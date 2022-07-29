from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10]) # shape=(10, )            
y = np.array([1,2,3,4,5,6,7,8,9,10]) # shape=(10, )

# train과 test비율을 8:2으로 분리하시오.
x_train =  x[:8] # [1 2 3 4 5 6 7 8]
x_test =  x[-2:] # [ 9 10]
y_train = y[:8]  # [1 2 3 4 5 6 7 8]
y_test = y[-2:]  # [ 9 10]

#2. 모델링
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=400, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss: ', loss) 
result = model.predict([11])
print('[11]의 예측값 : ', result)

# epochs=100, batch=1 [11]의 예측값 :  [[10.999258]]
# epochs=400, batch=1 [11]의 예측값 :  [[10.999997]]