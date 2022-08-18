import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping

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
    '../_data/image/brain/train/',
    target_size = (150, 150),                                                                       
    batch_size=1000,                                   
    class_mode='binary',                         
    shuffle=True,    
)   

xy_test = test_datagen.flow_from_directory(         
    '../_data/image/brain/test/',
    target_size=(150,150),
    batch_size=1000,
    class_mode='binary',                            
)  

#xy_train[0]     # batch값 만큼의 (개수)만큼의 데이터가 xy다 들어가 있다.
#xy_train[0][1]  # x값   (5개 묶음)
#xy_train[0][2]  # y값   (5개 묶음)

#print(xy_train[0][0].shape, xy_train[0][1])    # (160, 150, 150, 3)  (160,)  batch사이즈 엄청 주고 확인해보면 그냥 160이 한뭉텅이로 다 들어가있다.
#print(xy_test[0][0].shape, xy_test[0][1])     # (120, 150, 150, 3)  (120,) 상동.

np.save('./_save_npy/keras47_5_train_x.npy', arr=xy_train[0][0])    #arr 파라미터 이름이고 여기에 저장할 내용을넣는다. 
np.save('./_save_npy/keras47_5_train_y.npy', arr=xy_train[0][1])    # 
np.save('./_save_npy/keras47_5_test_x.npy', arr=xy_test[0][0])     # 
np.save('./_save_npy/keras47_5_test_y.npy', arr=xy_test[0][1])     # 

