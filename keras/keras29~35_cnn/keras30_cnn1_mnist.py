from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation
import numpy as np
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from pandas import get_dummies
from sklearn.preprocessing import OneHotEncoder
from icecream import ic

#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1)  # reshpe: 일렬로 만든 후 다시 나눠서 재배열하는 개념
x_test = x_test.reshape(10000, 28, 28, 1)  # 회색조(채널 1) 이미지

ic(np.unique(y_train, return_counts=True))  # pandas의 value.counts와 같은 기능
'''
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]                                           
'''
ic(y_train.shape)  # (60000,)

enco = OneHotEncoder(sparse=False)  # sparse=True가 디폴트로 Matrix를 반환한다. 원핫인코딩에서 필요한 것은 array이므로 sparse옵션에 False를 넣어준다.
y_train = y_train.reshape(-1,1)  # (60000, 1) --> 열에 맞춰 행은 가변적으로
y_train = enco.fit_transform(y_train) # (60000, 10)
ic(y_train[:5])  
'''
[[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], 
[0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
 '''

y_test = to_categorical(y_test)  # 원핫인코더 --> (60000, 10)
ic(y_test[:5]) 
'''
[[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]]
'''

#2. 모델링
model = Sequential()
model.add(Conv2D(10, kernel_size=(3, 3), input_shape=(28, 28, 1)))  
model.add(Conv2D(10, (3, 3), activation='relu'))
model.add(Conv2D(10, (2, 2), activation='relu'))
model.add(Conv2D(10, (2, 2), activation='relu'))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.4))
model.add(Dense(16))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras30_mnist_MCP.hdf5')
model.fit(x_train, y_train, epochs=1000, batch_size=1000, validation_split=0.2, callbacks=[es])
# model.save(f"./_save/keras30_save_mnist.h5")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
loss :  0.09150657057762146
accuracy :  0.9763000011444092
'''
