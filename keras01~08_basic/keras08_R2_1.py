from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt     
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score  

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]) # shape=(20, )
y = np.array([1,2,4,3,5,7,9,9,8,12,13,17,12,14,21,14,11,19,23,25]) # shape=(20, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(15))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')     
model.fit(x_train, y_train, epochs=500, batch_size=1)

# 4. 평가, 예측 
# loss = model.evaluate(x_test, y_test) 
# print('loss : ', loss)

y_predict = model.predict(x_test) # y의 예측값은 x의 테스트값에 wx + b 

r2 = r2_score(y_test, y_predict) # 계측용 y_test값과, y예측값을 비교한다.
print('r2스코어 : ', r2)

plt.scatter(x, y) # scatter 점
plt.plot(x_test, y_predict, color='red') # plot 선
plt.show() 