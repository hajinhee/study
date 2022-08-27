from tensorflow.keras import datasets
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from icecream import ic
from sklearn.metrics import accuracy_score  
# import warnings
# warnings.filterwarnings(action='ignore')  

#1. 데이터
(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(    
    rescale=1./255,                    
    horizontal_flip=True,               
    # vertical_flip=True,                                      
    width_shift_range=0.1,            
    height_shift_range=0.1,   
    # rotation_range=5,               
    zoom_range=0.1,                 
    # shear_range=0.7,                    
    fill_mode='nearest'        
)

test_datagen = ImageDataGenerator(
    rescale=1./255    # 테스트 데이터는 스케일링만(원본으로 사용)
    )

augment_size = 40000

randidx = np.random.randint(x_train.shape[0], size=augment_size)

x_augmented = x_train[randidx].copy()     
y_augmented = y_train[randidx].copy()      

# 입력 데이터 4차원 변환
x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)  # 입력데이터 4차원 변환
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# 카피한 데이터 증폭
xy_train = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=32,    
    shuffle=False
)

xy_test = test_datagen.flow(
    x_test, y_test, batch_size=32
)

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28, 28, 1)))
model.add(Conv2D(64, (2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])  # sparse_categorical_crossentropy: one_hot_encoding 과정 생략 가능
model.fit_generator(xy_train, epochs=10, steps_per_epoch=len(xy_train))   # fit_generator: 배치 단위로 생산한 데이터에 대해서 모델 학습

#4. 훈련, 평가
loss = model.evaluate_generator(xy_test)  # evaluate_generator: 배치 단위로 생산한 데이터에 대해서 모델 평가
ic(loss[0], loss[1])

y_pred = model.predict_generator(x_test)  # predict_generator: 배치 단위로 생산한 데이터에 대해서 모델 예측
ic(y_pred) 
'''
[[0., 0., 0., ..., 0., 1., 0.],
[0., 0., 1., ..., 0., 0., 0.],
[0., 1., 0., ..., 0., 0., 0.],
...,
[0., 0., 0., ..., 0., 1., 0.],
[0., 1., 0., ..., 0., 0., 0.],
[0., 0., 0., ..., 0., 1., 0.]]
'''
y_pred = np.argmax(y_pred, axis=1)  # one_hot_encoding 된 데이터 다시 int로 변환
ic(y_pred) 
'''
[8, 2, 1, ..., 8, 1, 8]
'''
accuracy = accuracy_score(y_test, y_pred)
ic(accuracy)  # accuracy: 0.5818
