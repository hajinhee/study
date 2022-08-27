import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,  # 좌우반전
    vertical_flip=True,  # 상하반전
    width_shift_range=0.1,  # 좌우이동
    height_shift_range=0.1,  # 상하이동
    rotation_range=5,  # 회전
    zoom_range=1.2,  # 확대
    shear_range=0.7,  # 기울기
    fill_mode='nearest'    
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    'dacon/seoul_landmark/data/train/',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',
    shuffle=True,
)       # Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary'
)       # Found 120 images belonging to 2 classes.

print(xy_train[0][0].shape)  # x_data (5, 150, 150, 3) --> (배치사이즈, 가로이미지, 세로이미지, 채널)
print(xy_train[0][1].shape)  # y_data (5,) --> (배치사이즈)

print(type(xy_train))  # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))  # <class 'tuple'>
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

#2. 모델
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3), activation='relu'))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit_generator(xy_train, epochs=30, steps_per_epoch=32,  # 전체데이터/batch_size = 160/5 = 32
                    validation_data=xy_test,
                    validation_steps=24,  # 전체 데이터/배치사이즈 = 120/5 = 24
                    )

