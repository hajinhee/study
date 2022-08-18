# LSTM
# LSTM은 RNN의 문제를 셀상태(Cell state)와 여러 개의 게이트(gate)를 가진 셀이라는 유닛을 통해 해결
# 이 유닛은 시퀀스 상 멀리 있는 요소를 잘 기억할 수 있도록 함
# 셀 상태는 기존 신경망의 은닉층이라고 생각할 수 있음
# 셀상태를 갱신하기 위해 기본적으로 3가지의 게이트가 필요함
# 3 개의 게이트와 1개의 cell state가 있어서 4배가 됨.
# sigmoid와 tanh 적절히 사용.

# Forget, input, output 게이트
# Forget : 이전 단계의 셀 상태를 얼마나 기억할 지 결정함. 0(모두 잊음)과 1(모두 기억) 사이의 값을 가짐
# input : 새로운 정보의 중요성에 따라 얼마나 반영할지 결정
# output : 셀 상태로부터 중요도에 따라 얼마나 출력할지 결정함

# 게이트는 가중치를 가진 은닉층으로 생각할 수 있음. 각 가중치는 sigmoid층에서 갱신되며 0과 1사이의 값을 가짐
# 이 값에 따라 입력되는 값을 조절하고, 오차에 의해 각 단계(time step)에서 갱신됨

# activation tanh Function

# sigmoid fuction을 보완하고자 나온 함수. 입력신호를 (−1,1) 사이의 값으로 normalization 해줌.
# 거의 모든 방면에서 sigmoid보다 성능이 좋음.
# 수식 : tanh(x) = e^x - e^-x / e^x + e^-x
#      d/dx tanh(x) = 1-tanh(x)^2

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

#x = np.array(1,2,3,4,5,6,7)
x = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]])     

y = np.array([5,6,7,8])                             
        
x = x.reshape(4,2,2)

#2. 모델구성
model = Sequential()
model.add(LSTM(32,input_shape=(2,2)))   # input_shape는 행 빼고 들어가서 형태는 -1차원이 된다.
model.add(Dense(10))        
model.add(Dense(8))                 
model.add(Dense(4))                 
model.add(Dense(2))                 
model.add(Dense(1))                         


#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam') #mae도있다.
es = EarlyStopping(monitor="loss", patience=500, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x,y, epochs=10000, batch_size=1, callbacks=[es])  

#4. 평가, 예측

model.evaluate(x,y)
y_pred = np.array([5,6,7,8]).reshape(1,2,2)
result = model.predict(y_pred)  
print(result)

