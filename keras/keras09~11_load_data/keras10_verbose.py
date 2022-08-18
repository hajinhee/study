from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.metrics import r2_score  
import time

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

start = time.time()
model.fit(x, y, epochs=500, batch_size=2, verbose=0) # verbose: 훈련 공개 여부, 0=안보여줌 / 1=다보여줌 / 2=loss까지 / 3=훈련횟수만
end = time.time() - start
print("걸린시간 : ", end) 

#4. 평가, 예측 
loss = model.evaluate(x, y) 
print('loss : ', loss) # 훈련 loss 값과 평가 loss 값의 차이가 적어야 한다. 
y_predict = model.predict(x) 
r2 = r2_score(y, y_predict) 
print('r2스코어 : ', r2)

# epochs=100, batch=1 [loss] :  0.40023642778396606,  [r2스코어] :  0.7998817329787926
# epochs=300, batch=1 [loss] :  0.3807086646556854,  [r2스코어] :  0.809645658304845
# epochs=500, batch=2 [loss] :  0.3814261257648468,  [r2스코어] :  0.8092869324567914


