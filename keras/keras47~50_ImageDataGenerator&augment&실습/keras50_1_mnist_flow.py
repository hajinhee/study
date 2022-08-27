from typing import Tuple
from tensorflow.keras import datasets
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from icecream import ic
# import warnings
# warnings.filterwarnings(action='ignore')

#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_datagen = ImageDataGenerator(  
    rescale=1./255, 
    horizontal_flip = True,
    # vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    # rotation_range= 5,
    zoom_range = 0.1,
    # shear_range = 0.7,
    fill_mode= 'nearest')

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_augmented = x_train[randidx].copy()  
y_augmented = y_train[randidx].copy() 

# 입력데이터 4차원 변환
x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 증폭 데이터 생성
xy_train = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size, 
    shuffle= False
    ) 

# 기존 데이터와 증폭 데이터 결합
x = np.concatenate((x_train, xy_train[0][0]))  # (100000, 28, 28, 1)
y = np.concatenate((y_train, xy_train[0][1]))

#2. 모델링
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28, 28, 1)))
model.add(Conv2D(64, (2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])  # sparse_categorical_crossentropy -> 원핫인코딩 생략 가능
model.fit(x, y, epochs=10, steps_per_epoch=1000)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss[0], loss[1])  # loss[0]: 0.37550997734069824, loss[1]: 0.8967000246047974

y_predict = model.predict(x_test)  # 원핫인코딩 상태
y_predict = np.argmax(y_predict, axis=1)  # int 상태
accuracy = accuracy_score(y_test, y_predict)
ic(accuracy)  # accuracy: 0.8967