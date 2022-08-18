from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception,VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings, os
# warnings.filterwarnings(action='ignore')

path = 'D:\_data\image_classification\\rps/'

# 나중에 사진 변환하고 저장할때 비교를 쉽게하기 위해 기존 사진들의 이름을 따서 저장.
paper = os.listdir(path+'paper')         
rock = os.listdir(path+'rock')                  
scissors = os.listdir(path+'scissors')                  

# print(len(paper),len(rock),len(scissors))    # 840 840 840
# 딱히 증폭시켜서 장수의 밸런스를 맞춰줄 필요는 없을듯 하다.

# save 구간 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
img_datagen = ImageDataGenerator(rescale= 1/255.,validation_split=0.2)

rps_train = img_datagen.flow_from_directory(      
    path,
    target_size = (200, 200),                                                                       
    batch_size=100000,                                   
    class_mode='categorical',  
    classes= ['paper','rock','scissors'],
    subset = 'training'        
)   

rps_test = img_datagen.flow_from_directory(         
    path,
    target_size=(200,200),
    batch_size=1000000,
    class_mode='categorical',  
    classes= ['paper','rock','scissors'],
    subset='validation'                              
) 

np.save(path + 'rps_train_x', arr=rps_train[0][0])    
np.save(path + 'rps_train_y', arr=rps_train[0][1])    
np.save(path + 'rps_test_x', arr=rps_test[0][0])      
np.save(path + 'rps_test_y', arr=rps_test[0][1]) 
'''

# load구간 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

x_train = np.load(path + 'rps_train_x.npy')      
y_train = np.load(path + 'rps_train_y.npy')     
x_test = np.load(path + 'rps_test_x.npy')       
y_test = np.load(path + 'rps_test_y.npy')
# print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) # (2016, 200, 200, 3) (2016, 3) (504, 200, 200, 3) (504, 3)

model_list = [VGG19(weights='imagenet', include_top=False, input_shape=(200,200,3),pooling='max',classifier_activation='softmax'),
              Xception(weights='imagenet', include_top=False, input_shape=(200,200,3),pooling='max',classifier_activation='softmax')]

for model in model_list:
    print(f"모델명 : {model.name}")
    TL_model = model
    TL_model.trainable = True
    model = Sequential()
    model.add(TL_model)    
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    
    optimizer = Adam(learning_rate=0.0001)  # 1e-4     
    lr=ReduceLROnPlateau(monitor= "val_acc", patience = 3, mode='max',factor = 0.1, min_lr=0.00001,verbose=False)
    es = EarlyStopping(monitor ="val_acc", patience=10, mode='max',verbose=1,restore_best_weights=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics='acc')
    
    start = time.time()
    model.fit(x_train,y_train,batch_size=20,epochs=1000,validation_split=0.2,callbacks=[lr,es], verbose=1)
    end = time.time()
    
    loss, Acc = model.evaluate(x_test,y_test,batch_size=20,verbose=False)
    
    print(f"Time : {round(end - start,4)}")
    print(f"loss : {round(loss,4)}")
    print(f"Acc : {round(Acc,4)}")

'''
모델명 : vgg19
Time : 203.1401
loss : 0.0001
Acc : 1.0

모델명 : xception
Time : 204.2005
loss : 0.0014
Acc : 1.0
'''