
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score     

#1. 데이터
x = np.array(range(100)) # shape=(100, )
y = np.array(range(1, 101)) # shape=(100, )

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)


#2. 모델구성
model = Sequential()
model.add(Dense(444, input_dim=1))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(10))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(10))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(444))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  # mean squared error , 평균제곱오차    
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측 
# loss = model.evaluate(x_test, y_test)
# print('loss : ', loss)

y_predict = model.predict(x_test) 

r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)


# epochs=100, batch=1  [r2스코어] :  -62.45789705624442

