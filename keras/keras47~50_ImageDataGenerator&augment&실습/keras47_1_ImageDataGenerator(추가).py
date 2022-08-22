import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator    
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

#1. 데이터로드 및 전처리
train_datagen = ImageDataGenerator(    
    rescale=1./255,                    
    horizontal_flip=True,               
    vertical_flip=True,                                      
    width_shift_range=0.1,            
    height_shift_range=0.1,   
    rotation_range=5,               
    zoom_range=1.2,                 
    shear_range=0.7,                    
    fill_mode='nearest'          
)

test_datagen = ImageDataGenerator(
    rescale=1./255                      
)

xy_train = train_datagen.flow_from_directory(      
    'dacon/seoul_landmark/data/train/',
    target_size=(150, 150),                                                                       
    batch_size=1,                                   
    class_mode='categorical',                         
    shuffle=True,               # 안의 내용을 섞어줌
    seed=66,                    # randomstate와 마찬가지로 랜덤값 고정. 변환할 정도를 랜덤셔플하게 섞어주겠다.
    color_mode='grayscale',     # 흑백과 컬러를 지정해줄수있다.
    save_to_dir='../_temp/'     # 데이터를 출력해서 보여준다.
)   

xy_test = test_datagen.flow_from_directory(         
    'dacon/seoul_landmark/data/test/',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',                            
)  

print(len(xy_train))   
print(xy_train[0])  # 배치사이즈로 1개씩 묶인 xy데이터
print(xy_train[0][0])  # xy_train의 첫번째 묶음의 첫번째 = x
print(xy_train[0][1])  # xy_train의 첫번째 묶음의 두번째 = y
print(xy_train[0][0].shape, xy_train[0][1].shape)
