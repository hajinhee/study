# import tensorflow as tf
from tensorflow.keras.models import Sequential # 시퀀셜 모델과
from tensorflow.keras.layers import Dense # 덴스 레이어를 쓸수있다. 
import numpy as np

#1. 데이터 정제해서 값 도출
x =  np.array([1,2,3]) # 입력 데이터, shape=(3, 1)
y =  np.array([1,2,3]) # 라벨

#2. 모델구성
model = Sequential() # 순차 모델
model.add(Dense(1, input_dim=1))  # 입력 = 벡터 1개, 출력 = 1개

#3. 컴파일 
model.compile(loss='mse', optimizer='adam') # 오차함수는 mse(평균제곱오차), optimizer는 adam을 사용한다. 85점 이상이면 쓸만하다.

#4. 훈련
model.fit(x, y, epochs=4200, batch_size=1) # epochs은 전체 데이터 훈련 횟수, batch_size는 한 번에 훈련시키는 데이터량

#4. 평가, 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)
result = model.predict([3]) # 새로운 x값을 predcit한 결과 
print('3의 예측값 : ', result)