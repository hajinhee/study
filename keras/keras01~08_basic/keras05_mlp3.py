import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([range(10), range(21,31), range(201,211)])
# print(x)
'''
[[  0   1   2   3   4   5   6   7   8   9] 
 [ 21  22  23  24  25  26  27  28  29  30] 
 [201 202 203 204 205 206 207 208 209 210]]
'''
# print(x.shape)  
'''
(3, 10)
'''
x = np.transpose(x) # shape=(10, 3)
y = np.array( [[1,2,3,4,5,6,7,8,9,10],
               [1,1.1,1.2,1.3,1.4,1.5,
                1.6,1.5,1.4,1.3],
               [10,9,8,7,6,5,4,3,2,1]])  # shape=(3, 10)
y = np.transpose(y) # shape=(10, 3)

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))
model.add(Dense(11))
model.add(Dense(8))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(12)) 
model.add(Dense(3))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=950, batch_size=1)

# 4. 평가 , 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)
result = model.predict([[ 9, 30, 210]])
print('[ 9, 30, 210]의 예측값 : ', result)

# 실제값: 10, 1.3, 1

# epochs=1000, batch=1  [ 9, 30, 210]의 예측값 :  [[10.015492   1.4906144  1.0629896]]
# epochs=1000, batch=3  [ 9, 30, 210]의 예측값 :  [[10.0619755  1.4918437  0.9049344]]
# epochs=1500, batch=1 [ 9, 30, 210]의 예측값 :  [[10.074265    1.4946842   0.86705387]]
# epochs=800, batch=1 [ 9, 30, 210]의 예측값 :  [[9.869738   1.5099006  0.88980794]]
