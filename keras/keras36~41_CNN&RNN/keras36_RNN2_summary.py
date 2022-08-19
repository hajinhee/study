import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from icecream import ic

#1. 데이터
x = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6], [4,5,6,7]])  # (4, 4) 2d
'''
[[1, 2, 3, 4],
[2, 3, 4, 5],
[3, 4, 5, 6],
[4, 5, 6, 7]]
'''
y = np.array([4, 5, 6, 7])  # (4, ) 1d  --> y_pred = (1, 1, 4)                        
x = x.reshape(4, 1, 4)  # (4, 1, 4) 3d  
'''
[[[1, 2, 3, 4]],

[[2, 3, 4, 5]],

[[3, 4, 5, 6]],

[[4, 5, 6, 7]]]

RNN은 2D 텐서가 아니라 3D 텐서를 입력을 받는다. 
즉, 위에서 만든 2D 텐서를 3D 텐서로 변경한다. 이는 배치 크기 1을 추가해 해결한다.
(batch_size(행), timesteps(또는 input_length), input_dim(열))에 해당되는 (4, 1, 4)의 크기를 가지는 3D 텐서가 생성되었다. 
batch_size는 한 번에 RNN이 학습하는 데이터의 양을 의미하지만, 여기서는 샘플이 1개 밖에 없으므로 batch_size는 1이다.
'''

# 2. 모델구성
model = Sequential()
model.add(SimpleRNN(10, input_shape=(1, 4)))  # 추가 인자를 사용할 때
# model.add(SimpleRNN(10, input_length=1, input_dim=4))  # 다른 표기
model.add(Dense(4))                 
model.add(Dense(2))                 
model.add(Dense(1))                         
model.summary()

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss', patience=500, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x, y, epochs=100, batch_size=1, callbacks=[es])  

#4. 평가, 예측
model.evaluate(x, y)
y_pred = np.array([5,6,7,8]).reshape(1, 1, 4)  # (1, 1, 4) 3d
result = model.predict(y_pred)    
ic(result)


