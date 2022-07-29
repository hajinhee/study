from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score 

#1. 데이터
x = np.array([1,2,3,4,5]) # shape=(5, )
y = np.array([1,2,4,3,5]) # shape=(5, )

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  
model.fit(x,y,epochs=500, batch_size=1)

#4. 평가, 예측 
#loss = model.evaluate(x,y)
#print('loss : ', loss)

y_predict = model.predict(x) 
r2 = r2_score(y, y_predict) 
print('r2스코어 : ', r2)

# epochs=500, batch=1  [r2스코어] :  0.8098438458340069