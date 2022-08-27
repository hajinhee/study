from tensorflow.keras import datasets
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pandas import get_dummies
from tensorflow.keras.utils import to_categorical 
from icecream import ic
from sklearn.metrics import accuracy_score

#1. 데이터 로드 및 정제
(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()
ic(x_train.shape)  # (60000, 28, 28)
ic(y_train.shape)  # (60000,)     
ic(x_test.shape)  # (10000, 28, 28) 

train_datagen = ImageDataGenerator(    
    rescale=1./255,  # 스케일링                 
    horizontal_flip=True,  # 수평, 열방향 좌우반전
    # vertical_flip=True,  # 수직, 행방향 상하반전
    width_shift_range=0.1,  # 좌우이동           
    height_shift_range=0.1,  # 상하이동
    # rotation_range=5,  # 회전
    zoom_range=0.1,  # 확대
    # shear_range=0.7,  # 기울기
    fill_mode='nearest' 
)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size)  # (60000, 40000) --> 60000개 안에서 40000개의 랜덤 정수 값 출려
ic(randidx.shape)  # (40000,)

x_augmented = x_train[randidx].copy()  # (40000, 28, 28) --> 랜덤으로 뽑힌 해당 인덱스의 값 복사
y_augmented = y_train[randidx].copy()  # (40000,)   

# 입력데이터 4차원 변환
x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)
x_train = x_train.reshape(60000, 28, 28, 1)  
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  

x_augmented = train_datagen.flow(
    x_augmented, np.zeros(augment_size),  # 임의의 값?
    batch_size=augment_size, shuffle=False
).next()[0]  # x값만 출력

# 제너레이터 데이터와 기존 데이터 결합
x_train = np.concatenate((x_train, x_augmented))  
y_train = np.concatenate((y_train, y_augmented))

# 원핫인코딩
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델링
model = Sequential()
model.add(Conv2D(10, kernel_size=(2, 2), input_shape=(28, 28, 1)))  
model.add(Conv2D(10, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=1, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, callbacks=[es])
model.save('./_save/keras30_2_save_model.h5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=10)
ic(loss[0], loss[1])  # loss[0]: 0.3246285319328308, loss[1]: 0.8865000009536743

y_pred = model.predict(x_test, batch_size=10)
y_pred = np.argmax(y_pred, axis=1) 
y_test = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_pred, y_test)
ic(accuracy)  # accuracy: 0.8591

