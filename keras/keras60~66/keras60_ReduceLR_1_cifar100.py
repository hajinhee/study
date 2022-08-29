import numpy as np,time,warnings
from tensorflow.keras.datasets import mnist, cifar100
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from icecream import ic

#1. load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
ic(x_train.shape)  # (50000, 32, 32, 3)
x_train = x_train.reshape(50000,32,32,3)/255.  # (50000, 32, 32, 3) --> 4차원 변환 및 정규화
x_test = x_test.reshape(10000,32,32,3)/255.  # (10000, 32, 32, 3)

#2. modeling
model = Sequential()
model.add(Conv2D(128, (2,2), padding='valid', activation='relu', input_shape=(32, 32, 3)))  
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())      
# model.add(GlobalAveragePooling2D())     
'''
[AveragePooling2D] 
 pool_size 크기의 윈도우가 stride만큼 이동하면서 윈도우 값의 평균을 출력하는 함수
 ex) 64x64x256의 AveragePooling2D(2,2, stride=2)의 출력은 32x32x256
[GlobalAveragePooling2D]
 n개의 채널을 평균하며 하나의 값으로 표현
 ex) 64x64를 평균한 값 256개가 나옴
'''
model.add(Dense(100, activation='softmax'))  # 다중분류

#3. compile, train
learning_rate = 0.01  
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, min_lr=0.0001, factor=0.5)  # 값이 갱신이 안되는 순간 lr 0.5배(*0.5) 감소  
'''
모델이 돌아가다가 값이 5번 이내에 갱신되지 않으면 lr감소, 다시 5번 이내에 갱신되지 않으면 lr감소, 
그러다 15번째는 EarlyStopping 되기 때문에 EarlyStopping 과 ReduceLROnPlateau 사이의 값 조정이 필요하다.
'''

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[reduce_lr, es])
end = time.time() - start

#4. evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size=32)  # 평가 속도를 높이기 위해 배치 크기를 갖는다. 
ic(learning_rate, round(loss, 4), round(acc, 4), f'걸린시간: {round(end, 4)}')

'''
[Flatten]
learning_rate: 0.01
round(loss, 4): 3.4252
round(acc, 4): 0.2195
걸린시간:  1071.5032

[GlobalAveragePooling]


'''
