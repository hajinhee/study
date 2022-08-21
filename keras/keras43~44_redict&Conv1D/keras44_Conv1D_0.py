import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Bidirectional, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from pandas import get_dummies

'''
시계열 데이터처리에 많이 사용되는 '1D convolution'
2D convolution이 가로,세로로 모두 이동하면서 output이 계산되는 것과 다르게 
1D convolution 연산은 가로로만 이동하면서 output을 계산해낸다.
- Conv1d(out_channels=, kernel_size=, ... )
- out_channels: 내가 output으로 내고싶은 dimension
- kernel_size: time step을 얼마만큼 볼 것인가(=frame size = filter size)
'''

def split_x(dataset, size):                            
    aaa = []                                            
    for i in range(len(dataset) - size + 1):            
        subset = dataset[i : (i + size)]              
        aaa.append(subset)                            
    return np.array(aaa) 

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
x = x_train  # (50000, 32, 32, 3)
print(np.unique(y_train, return_counts=True))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000] -> 다중분류
y = get_dummies(y_train.reshape(len(y_train)))  # (50000, 10) 
y_test = get_dummies(y_test.reshape(len(y_test))) 
x = x.reshape(len(x), 16, 192)  # (50000, 16, 192)   
x_test = x_test.reshape(len(x_test), 16, 192) 

#2. 모델구성
model = Sequential() 
model.add(Conv1D(10, 7, input_shape=(16, 192)))  # Conv1D -> 3차원 데이터 입력
model.add(Conv1D(8, 8))
model.add(Flatten())
model.add(Dense(10))                                      
model.add(Dense(8))                 
model.add(Dense(4))                 
model.add(Dense(2))                 
model.add(Dense(1))                     
model.summary()

'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv1d (Conv1D)             (None, 10, 10)            13450

 conv1d_1 (Conv1D)           (None, 3, 8)              648

 flatten (Flatten)           (None, 24)                0

 dense (Dense)               (None, 10)                250

 dense_1 (Dense)             (None, 8)                 88

 dense_2 (Dense)             (None, 4)                 36

 dense_3 (Dense)             (None, 2)                 10

 dense_4 (Dense)             (None, 1)                 3

=================================================================
Total params: 14,485
Trainable params: 14,485
Non-trainable params: 0
_________________________________________________________________
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor='loss', patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x, y, epochs=10, batch_size=10000, callbacks=[es])  

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('[loss]: ', round(loss, 4))

'''
[loss]:  5813.2778
'''
