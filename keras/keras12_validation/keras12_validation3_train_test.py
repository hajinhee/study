from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split # 싸이킷런, 싸이킷런 

#1. 데이터
x = np.array(range(1,17)) # shape=(17, )
y = np.array(range(1,17)) # shape=(17, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.625, random_state=66) 
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=49)

print(x_train) # [10  3 15  6 12 11  4  8 13  5]
print(y_train) # [10  3 15  6 12 11  4  8 13  5]
print(x_val, y_val) # [ 2  7 14] [ 2  7 14]
print(x_test) # [16  9  1]
print(y_test) # [16  9  1]


# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13])
# y_test = np.array([11,12,13])
# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)