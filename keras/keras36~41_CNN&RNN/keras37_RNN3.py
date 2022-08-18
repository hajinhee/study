import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
x = np.array([[1,2,3],                               #array는 행렬을 의미! 모든 연산은 numpy로 한다. 
             [2,3,4],                                #x에서 [1,2,3]으로 짜른 범위는 timesteps;'3', 그 안의 1은 feature;'1'
             [3,4,5],
             [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape, y.shape)   # (4, 3) (4,)

# input_shape = (batch_size, timesteps, feature)
# input_shape = (행, 열, 몇개씩 짜르는지!!!)
x = x.reshape(4, 3, 1)

#2. 모델구성
model = Sequential()
#model.add(SimpleRNN(10, input_shape=(3,1)))                    
#model.add(SimpleRNN(units=10, input_shape=(3,1)))             ======> SimpleRNN은 세가지 방법으로 쓸 수 있다.
model.add(SimpleRNN(10, input_length=3, input_dim=1))   

model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()
'''
* SimpleRNN 에서의 "output"
units: Positive integer(양수), dimensionality of the output space
*conv2D 에서의 "output"
filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
'''
# 파라미터 개수 구조 확인 
# https://velog.io/@cateto/Deep-Learning-vanila-RNN%EC%97%90%EC%84%9C-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0%EC%9D%98-%EA%B0%9C%EC%88%98-%EA%B5%AC%ED%95%98%EA%B8%B0


'''
#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')      # optimizer는 loss를 최적화한다
model.fit(x, y, epochs=100, batch_size=1)
#4. 평가, 예측
model.evaluate(x, y)
x2 = np.array([5,6,7]).reshape(1,3,1)
result = model.predict(x2)
print(result)
'''