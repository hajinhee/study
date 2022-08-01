import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10)]) # shape=(1, 10)
'''
[[0 1 2 3 4 5 6 7 8 9]]
'''
x = np.transpose(x)  # shape=(10, 1)
'''
[[0]
 [1]
 [2]
 [3]
 [4]
 [5]
 [6]
 [7]
 [8]
 [9]]
 '''
y = np.array( [[1,2,3,4,5,6,7,8,9,10],
               [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],
               [10,9,8,7,6,5,4,3,2,1]]) # shape=(3, 10)
y = np.transpose(y) # shape=(10, 3)
'''
[[ 1.   1.  10. ]
 [ 2.   1.1  9. ]
 [ 3.   1.2  8. ]
 [ 4.   1.3  7. ]
 [ 5.   1.4  6. ]
 [ 6.   1.5  5. ]
 [ 7.   1.6  4. ]
 [ 8.   1.5  3. ]
 [ 9.   1.4  2. ]
 [10.   1.3  1. ]]
'''

# 2. 모델구성 
model = Sequential()
model.add(Dense(10, input_dim=1))  # 입력데이터(x)의 열 개수를 넣는다.
model.add(Dense(5))
model.add(Dense(11))
model.add(Dense(8))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(5))
model.add(Dense(12)) 
model.add(Dense(3))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=300, batch_size=1) 

# 4. 평가 , 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)
result = model.predict([9])  
print('[9]의 예측값 : ', result)

# 실제값: 10, 1.3, 1
# epochs=1000, batch=1 [9]의 예측값 :  [[9.779481  1.5131357 0.9064951]]
# epochs=500, batch=1 [9]의 예측값 :  [[9.915947  1.4517137 1.0204992]]
# epochs=1500, batch=3 [9]의 예측값 :  [[9.601834  1.4592481 0.6318351]]
# epochs=300, batch=1 [9]의 예측값 :  [[9.986538  1.5335737 0.9730082]]

