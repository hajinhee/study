import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from icecream import ic
# import warnings
# warnings.filterwarnings(action='ignore')

#1. 데이터 로드 및 전처리
(x_train, y_train),(x_test,y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape)  #(50000, 32, 32, 3) (50000, 1)  
# print(x_test.shape, y_test.shape)  #(10000, 32, 32, 3) (10000, 1)

ic(np.unique(y_train, return_counts=True))  
'''
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8),
array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]  --> 다중분류
'''

train_augment_datagen = ImageDataGenerator(    
    horizontal_flip=True,  
    rotation_range=3,       
    width_shift_range=0.3, 
    height_shift_range=0.3, 
    zoom_range=(0.3),       
    fill_mode='nearest',                   
)
all_datagen = ImageDataGenerator(
    rescale=1./255.,
    validation_split=0.2 
)

augment_size = 100000-x_train.shape[0]    # 총 10만개 데이터를 만들기 위해 augment_size 지정

# 기존 데이터 카피 
x_augmented = x_train.copy()      
y_augmented = y_train.copy() 

# 카피 데이터 증폭
x_augmented = train_augment_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size,
).next()[0]  # x_data

# 기존 데이터와 가피 데이터 결합
real_x_train = np.concatenate((x_train, x_augmented))   
real_y_train = np.concatenate((y_train, y_augmented))

xy_train_train = all_datagen.flow(
    real_x_train, real_y_train,
    batch_size=100,
    shuffle=True,
    seed=66,
    subset='training'  # set training data
) 
xy_train_val = all_datagen.flow(
    real_x_train, real_y_train,
    batch_size=100,
    shuffle=True,
    seed=66,
    subset='validation'  # set validation data
)

xy_test = all_datagen.flow(
    x_test, y_test,
    batch_size=100  
)

#2. 모델링
model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3),strides=1, padding='valid', input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(2,2))                                                                              
model.add(Conv2D(10, kernel_size=(2,2), strides=1, padding='same', activation='relu'))                      
model.add(MaxPooling2D(3,3))                                                                                
model.add(Conv2D(10, (2,2), activation='relu'))                                                        
model.add(MaxPooling2D(2,2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) 
es = EarlyStopping(monitor='val_acc', patience=50, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit_generator(xy_train_train, epochs=10, steps_per_epoch=len(xy_train_train), validation_data=xy_train_val, validation_steps=len(xy_train_val), callbacks=[es])

#4. 평가, 예측
loss = model.evaluate_generator(xy_test, steps=len(xy_test))
print('[loss] : ', loss[0])
print('[accuracy] : ', loss[1])

y_pred = model.predict_generator(x_test)
y_pred = np.argmax(y_pred, axis=1)        
acc = accuracy_score(y_test, y_pred)
print('[accuracy_score] : ', acc)

'''
[loss] :  1.804343342781067
[accuracy] :  0.11500000208616257
[accuracy_score] :  0.2166
'''
