import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터 로드 및 전처리
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
    rescale=1./255  # 테스트 데이터는 이미지증폭을 하지 않는다. 원본 사용 
)

xy_train = train_datagen.flow_from_directory(      
    'dacon/seoul_landmark/data/train/',
    target_size = (150, 150),
    batch_size=1000,
    class_mode='binary', 
    shuffle=True,    
)   

xy_test = test_datagen.flow_from_directory(         
    'dacon/seoul_landmark/data/test/',
    target_size=(150, 150),
    batch_size=1000,
    class_mode='binary',  # 테스트데이터는 셔플X
)  

xy_train[0]  # batch_size 만큼의 xy데이터가 들어있다.
xy_train[0][0]  # batch_size 만큼의 x데이터가 들어있다.
xy_train[0][1]  # batch_size 만큼의 y데이터가 들어있다.

print(xy_train[0][0].shape, xy_train[0][1])  # (160, 150, 150, 3)  (160, )  -->  batch_size 크게 주고 확인해보면 160이 한뭉텅이로 들어있다.
print(xy_test[0][0].shape, xy_test[0][1])  # (120, 150, 150, 3)  (120, ) 


np.save('./_save_npy/keras47_5_train_x.npy', arr=xy_train[0][0])   
np.save('./_save_npy/keras47_5_train_y.npy', arr=xy_train[0][1])    
np.save('./_save_npy/keras47_5_test_x.npy', arr=xy_test[0][0])     
np.save('./_save_npy/keras47_5_test_y.npy', arr=xy_test[0][1])     
