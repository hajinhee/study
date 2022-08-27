from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten ,MaxPooling2D
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from icecream import ic
from sklearn.metrics import accuracy_score 

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip= False,
    width_shift_range = 0.1,
    height_shift_range= 0.1,
    # rotation_range= 5,
    zoom_range = 0.1,              
    # shear_range=0.7,
    fill_mode = 'nearest'          
    )

augment_size = 50000
randidx = np.random.randint(x_train.shape[0], size=augment_size) 

# 랜덤 데이터 복사
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

# 입력 데이터 4차원 변환
x_augmented = x_augmented.reshape(50000, 32, 32, 3)
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

# 카피한 데이터 증폭
xy_train = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size, 
    shuffle=False
    )

# 기존 데이터와 결합
x = np.concatenate((x_train, xy_train[0][0]))  
y = np.concatenate((y_train, xy_train[0][1]))

#2. 모델링
model = Sequential() 
model.add(Conv2D(7, kernel_size=(2,2), strides=1, padding='same', input_shape=(32, 32, 3))) 
model.add(MaxPooling2D())
model.add(Conv2D(5, (3,3), activation='relu'))                     
model.add(Dropout(0.2))
model.add(Conv2D(4, (2,2), activation='relu'))         
model.add(Flatten()) 
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=10, steps_per_epoch=1000)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss)  # [loss]: 3.468953847885132, [accuracy]: 0.18639999628067017

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1) 
accuracy = accuracy_score(y_test, y_predict)
print('[accuracy]: ', accuracy)  # [accuracy]:  0.1864