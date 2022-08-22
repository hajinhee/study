import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from sklearn.datasets import load_boston
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from icecream import ic


train_datagen = ImageDataGenerator(     # 함수선언 및 정의 
    rescale=1./255,                     # Minmax scale -> 이미지는 최소 최대값이 0~255 사이이므로 이 작업해주는게 곧 minmax작업과 같음
    horizontal_flip=True,               # 상하반전 -> mnist같은건 상하반전하면 안된다. 
    vertical_flip=True,                 # 좌우반전                                  
    width_shift_range=0.1,              # 좌우이동
    height_shift_range=0.1,             # 상하이동
    rotation_range=5,                   # 이미지회전값
    zoom_range=1.2,                     # 이미지확대
    shear_range=0.7,                    # 이미지기울기
    fill_mode='nearest'                 # "constant", "nearest", "reflect" 혹은 "wrap" 인풋 경계의 바깥 공간은 다음의 모드에 따라 다르게 채워진다.
)

test_datagen = ImageDataGenerator(
    rescale=1./255      
                )                       # 테스트 데이터는 증폭이나 변조를 하지않는다. 평가 데이터는 원래의 이미지를 사용해야 하기 때문에 

xy_train = train_datagen.flow_from_directory(    # 경로설정. 분류형식의 이미지 받으면 폴더형식에 맞춰서 세팅.
    'dacon/seoul_landmark/data/train/',
    target_size=(150, 150),                      # 데이터의 사이즈 크기를 넣는게 아니라 내가 원하는 데이터 사이즈를 지정
    batch_size=5,                                # batch_size 지정 -> image를 batch_size 단위로 묶어서 수치화해서 저장한다. 
    class_mode='binary',                         # 분류 방식 지정
    # 'categorical' : 2D one-hot 부호화된 라벨이 반환, 'binary' : 1D 이진 라벨이 반환, 'sparse' : 1D 정수 라벨이 반환, None : 라벨이 반환되지 않음      
    shuffle=True,    
)    

xy_test = test_datagen.flow_from_directory(   # 정의단계에서 세팅을 다 하고 flow~로 데이터 가져와서 xy분류해준다.
    'dacon/seoul_landmark/data/test/',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',                            
)  

ic(xy_train)

datasets = load_boston()
ic(datasets)           

ic(xy_train[0])       # y값이 y의 batch사이즈 개수만큼 나온다.
ic(xy_train[0][0])    # xy_train의 첫번째 묶음의 첫번째 = x
ic(xy_train[0][1])    # xy_train의 첫번째 묶음의 두번째 = y
ic(xy_train[0][0].shape, xy_train[0][1].shape)  # (5, 150, 150, 3)  (5,)  5개씩 묶었으므로 5, 150,150원래 사이즈 흑백처럼보이지만 컬러사진이었다 그래서 채널 3

ic(type(xy_train))        # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
ic(type(xy_train[0]))     # <class 'tuple'>
ic(type(xy_train[0][0]))  # <class 'numpy.ndarray'>
ic(type(xy_train[0][1]))  # <class 'numpy.ndarray'>

