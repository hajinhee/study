'''
지금까지 배운 기존의 Train & Test 시스템은 훈련 후 바로 실전을 치뤄서 실전에서 값이 많이 튀었다.
이걸 더 보완하기위해서 Train 후 validation(검증) 거치는 걸 1epoch로 하여 계속 반복하며 값(가중치)을 수정 후
여기서 validation에서 나오는 val_loss가 fit에 직접적으로 관여하지는 않지만 방향성을 제시해준다.
실전 Test를 치루는 방식으로 더 개선시킨다. 앞으로는 데이터를 받으면 train validation test 
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. 데이터 
x = np.array(range(1, 17)) # (17, )
y = np.array(range(1,17)) # (17, )
x_train = x[:10]
y_train = y[:10]
x_test = x[11:14]
y_test = y[11:14]
x_val =  x[-3:]
y_val =  y[-3:]


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mse', optimizer='adam')

#4. 훈련
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

#5. 평가
model.evaluate(x_test, y_test)

#6. 예측
y_predict = model.predict([17])
print("17의 예측값 : ", y_predict)