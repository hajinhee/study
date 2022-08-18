from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception,VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings, os
# warnings.filterwarnings(action='ignore')


path = 'D:\_data\image_classification\cat_dog'

# 나중에 사진 변환하고 저장할때 비교를 쉽게하기 위해 기존 사진들의 이름을 따서 저장.
# train_cats = os.listdir(path+'/training_set/dogs')         
# train_dogs = os.listdir(path+'/training_set/cats')          
# test_cats = os.listdir(path+'/test_set/cats')          
# test_dogs = os.listdir(path+'/test_set/dogs')          

# print(len(train_cats),len(train_dogs),len(test_cats),len(test_dogs))    # 4005 4000 1011 1012 
# 딱히 증폭시켜서 장수의 밸런스를 맞춰줄 필요는 없을듯 하다.

# save 구간 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

'''
img_datagen = ImageDataGenerator(rescale= 1/255.)

catdog_train = img_datagen.flow_from_directory(      
    path + '/training_set/',
    target_size = (100, 100),                                                                       
    batch_size=100000,                                   
    class_mode='binary',  
    classes= ['cats','dogs']        
)   

catdog_test = img_datagen.flow_from_directory(         
    path + '/test_set/',
    target_size=(100,100),
    batch_size=1000000,
    class_mode='binary',  
    classes= ['cats','dogs']                          
) 

np.save(path + '/training_set/catdog_train_x', arr=catdog_train[0][0])    
np.save(path + '/training_set/catdog_train_y', arr=catdog_train[0][1])    
np.save(path + '/test_set/catdog_test_x', arr=catdog_test[0][0])      
np.save(path + '/test_set/catdog_test_y', arr=catdog_test[0][1]) 
'''


# load구간 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

x_train = np.load(path + '/training_set/catdog_train_x.npy')      
y_train = np.load(path + '/training_set/catdog_train_y.npy')     
x_test = np.load(path + '/test_set/catdog_test_x.npy')       
y_test = np.load(path + '/test_set/catdog_test_y.npy')
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) # (8005, 100, 100, 3) (8005,) (2023, 100, 100, 3) (2023,)

model_list = [VGG19(weights='imagenet', include_top=False, input_shape=(100,100,3),pooling='max',classifier_activation='sigmoid'),
              Xception(weights='imagenet', include_top=False, input_shape=(100,100,3),pooling='max',classifier_activation='sigmoid')]
            # pooling의 역할. 전이학습 모델의 마지막레이어를 flatten넣을지 globalavarge 넣을지 고를 수 있다...
for model in model_list:
    print(f"모델명 : {model.name}")
    TL_model = model
    TL_model.trainable = True
    model = Sequential()
    model.add(TL_model)    
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    optimizer = Adam(learning_rate=0.0001)  # 1e-4     
    lr=ReduceLROnPlateau(monitor= "val_acc", patience = 3, mode='max',factor = 0.1, min_lr=0.00001,verbose=False)
    es = EarlyStopping(monitor ="val_acc", patience=15, mode='max',verbose=1,restore_best_weights=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='acc')
    
    start = time.time()
    model.fit(x_train,y_train,batch_size=50,epochs=1000,validation_split=0.2,callbacks=[lr,es], verbose=1)
    end = time.time()
    
    loss, Acc = model.evaluate(x_test,y_test,batch_size=50,verbose=False)
    
    print(f"Time : {round(end - start,4)}")
    print(f"loss : {round(loss,4)}")
    print(f"Acc : {round(Acc,4)}")
'''
모델명 : vgg19
Time : 1155.5356
loss : 0.4445
Acc : 0.9466

모델명 : xception
Time : 341.0822
loss : 0.193
Acc : 0.9199
'''
