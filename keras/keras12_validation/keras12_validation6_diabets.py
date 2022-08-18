from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from icecream import ic

#1. 데이터
datasets = load_diabetes()
x = datasets.data 
y = datasets.target

ic(x.shape) # (442, 10)
ic(x.shape) # (442, 10)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49)

#2. 모델링 
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
k = model.fit(x_train, y_train, epochs=500, batch_size=3, validation_split=0.1)

#4. 평가
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

#5. 예측
y_predict = model.predict(x_test) 
r2 = r2_score(y_test, y_predict) # 예측한 값과 실제 값 비교
print('r2스코어 : ', r2)

# epochs=100, batch=1  [loss] :  2117.7021484375  [r2스코어] :  0.6025567117885298
# epochs=300, batch=1  [loss] :  2149.5361328125  [r2스코어] :  0.5965821917378205
# epochs=500, batch=3  [loss] :  1818.087158203125  [r2스코어] :  0.6420323958216063
