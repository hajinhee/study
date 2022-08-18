# https://www.kaggle.com/c/dogs-vs-cats/data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

#1.데이터 로드 및 전처리

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
    '../_data/image/cat_dog/training_set/',
    target_size = (200, 200),                                                                       
    batch_size=10,                                   
    class_mode='binary',        # 여기서 이제 이진분류면 binary 다중분류면 categorical해주는듯.
    shuffle=True,    
)   

xy_test = test_datagen.flow_from_directory(         
    '../_data/image/cat_dog/test_set/',
    target_size=(200,200),
    batch_size=10,
    class_mode='binary',                            
)  

#print(len(xy_train))   5개씩 묶었늗네 1601이 나왔다 나머지 연산 포함해서 대략 8000~8005장 사이인 걸 알 수있다. batch_size를 높이자.
#print(len(xy_test))    2025장 정도.
#print(len(xy_train),len(xy_test))   #batch 20에 401 102    10에  801 203
#print(xy_train[0][0].shape)     #(20, 200, 200, 3)  채널 3 확인.
#print(xy_train[0][1].shape)      #(20,)


#2. 모델링
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same',input_shape=(200,200,3), activation='relu'))
model.add(MaxPool2D(2))                                                     # 50,50,32
model.add(Conv2D(16, (4,4), activation='relu'))                            # 72,72,16
model.add(Flatten())
model.add(Dense(36, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일,훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor = "acc", patience=50, mode='max',verbose=1,restore_best_weights=True)
model.fit_generator(xy_train,epochs=10000,steps_per_epoch=400,callbacks=[es])            
                     
#4. 평가,예측.

loss = model.evaluate_generator(xy_test)
print(loss)     # [0.6193896532058716, 0.6450815796852112]     loss와 acc

