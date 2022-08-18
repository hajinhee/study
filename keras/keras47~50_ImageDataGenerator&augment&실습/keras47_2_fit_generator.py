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
    batch_size=5,                                   
    class_mode='binary',                         
    shuffle=True,    
)   

xy_test = test_datagen.flow_from_directory(         
    '../_data/image/brain/test/',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',                            
)  

#2. 모델링

model = Sequential()
model.add(Conv2D(32, (2,2), padding='same',input_shape=(150,150,3), activation='relu'))
model.add(MaxPool2D(2))                                                     # 75,75,32
model.add(Conv2D(16, (4,4), activation='relu'))                            # 72,72,16
model.add(MaxPool2D(4))
model.add(Flatten())
model.add(Dense(36, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1,activation='sigmoid'))

#3. 컴파일,훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor = "val_acc", patience=50, mode='max',verbose=1,restore_best_weights=True)
#model.fit(xy_train[0][0], xy_train[0][1])
hist = model.fit_generator(xy_train,epochs=10000,steps_per_epoch=32, callbacks=[es],# 전체데이터개수 160개를 5배치로 묶어서 32묶음. 이걸 1epoch당 32만큼하겠다? 라고 지정해줌. 확인차의 의미
                    validation_data=xy_test,
                    validation_steps=4,             # 이게 뭔지 찾아보자.
                    )  

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-51])
print('val_loss : ', val_loss[-51])
print('acc : ', acc[-51])
print('val_acc : ', val_acc[-51])


import matplotlib.pyplot as plt

plt.plot(acc, color='red', marker='.',label='acc') 
plt.plot(val_acc, color='purple', marker='.',label='val_acc') 
plt.plot(loss, color='green', marker='.',label='loss') 
plt.plot(val_loss, color='blue', marker='.',label='val_loss') 
plt.ylabel('values')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()




