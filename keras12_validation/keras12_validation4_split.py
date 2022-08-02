from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split # 싸이킷런, 싸이킷런 

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

# train_test_split로 나누시오 10 3 3 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

print(x_train) # [10  3 15  6 12 11  4  8 13  5]
print(y_train) # [10  3 15  6 12 11  4  8 13  5]
print(x_test) # [16  9  1]
print(y_test) # [16  9  1]

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1,
           validation_split=0.3)
# validation_split=0.3 기능 하나로 그 전과 비슷하게 x_train을 다시 train과 validaion으로 알아서 비율까지 쪼개준다. 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)