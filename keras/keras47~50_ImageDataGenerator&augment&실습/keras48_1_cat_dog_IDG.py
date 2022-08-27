from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

#1.데이터 로드 및 전처리
train_datagen = ImageDataGenerator(    
    rescale=1./255,  # 스케일링
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
    rescale=1./255  # 테스트 이미지는 이미지증폭을 하지 않는다. --> 원본 그대로 사용
)

xy_train = train_datagen.flow_from_directory(      
    'keras/data/images/cat_or_dog/train/',
    target_size = (200, 200),  # 이미지 사이즈 
    batch_size=10,  # 배치사이즈
    class_mode='binary',  # 이진분류는 'binary' 다중분류는 'categorical' --> 이 때 클래스 개수만큼 폴더로 분리되어 있어야 한다. 
    shuffle=True,    
)   # Found 402 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(         
    'keras/data/images/cat_or_dog/test/',
    target_size=(200, 200),  # 이미지 사이즈
    batch_size=10,  # 배치 사이즈
    class_mode='binary',  # 이진분류면 'binary' 다중분류면 'categorical' --> 이 때 클래스 개수만큼 폴더로 분리되어 있어야 한다.
)   # 테스트 데이터에서는 셔플X
    # Found 202 images belonging to 2 classes.

print(len(xy_train))  # 41 --> 총 데이터 수/배치사이즈 
print(len(xy_test))  # 21 --> 총 데이터 수/배치사이즈 
print(xy_train[0][0].shape)  # (10, 200, 200, 3) --> (배치사이즈, 이미지가로, 이미지세로, 채널)
print(xy_train[0][1].shape)  # (10,) --> (배치사이즈)
print(xy_train[0][1])  # [0. 1. 1. 1. 0. 1. 1. 0. 0. 1.]

#2. 모델링
model = Sequential()
model.add(Conv2D(32, (2, 2), padding='same', input_shape=(200, 200, 3), activation='relu'))
model.add(MaxPool2D(2))                                   
model.add(Conv2D(16, (4, 4), activation='relu'))                   
model.add(Flatten())
model.add(Dense(36, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 'sigmoid' 이진분류

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor='accuracy', patience=50, mode='max', verbose=1, restore_best_weights=True)
model.fit_generator(xy_train, epochs=100, steps_per_epoch=41, callbacks=[es])            
                     
#4. 평가, 예측.
loss = model.evaluate_generator(xy_test)
print(' [loss]: ', loss[0], '\n', '[accuracy]: ', loss[1])    

'''
[loss]:  0.6956690549850464 
[accuracy]:  0.46039605140686035
'''