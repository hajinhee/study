from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score

#1. 데이터 로드 및 전처리

train_datagen = ImageDataGenerator(    
    rescale=1./255,                     
    horizontal_flip=True,             
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest', 
    validation_split=0.2        
)

test_datagen = ImageDataGenerator(
    rescale=1./255                      
)

xy_train = train_datagen.flow_from_directory(      
    'keras/data/images/cat_or_dog/train/',
    target_size = (200, 200),                                                                       
    batch_size=20,                                   
    class_mode='binary',        
    shuffle=True  
)   # Found 402 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(         
    'keras/data/images/cat_or_dog/test/',
    target_size=(200, 200),
    batch_size=20,
    class_mode='binary',                            
)   # Found 202 images belonging to 2 classes.

print(len(xy_train))   # 21
print(len(xy_test))    # 11
print(xy_train[0][0].shape)  # (20, 200, 200, 3)
print(xy_train[0][1].shape)  # (20,)

# np.save('keras/save/npy/keras48_1_train_x.npy', arr=xy_train[0][0])  # x_traom  
# np.save('keras/save/npy/keras48_1_train_y.npy', arr=xy_train[0][1])  # y_train
# np.save('keras/save/npy/keras48_1_test_x.npy', arr=xy_test[0][0])  # x_test
# np.save('keras/save/npy/keras48_1_test_y.npy', arr=xy_test[0][1])  # y_test

x_train = np.load('keras/save/npy/keras48_1_train_x.npy')  # (20, 200, 200, 3)    
y_train = np.load('keras/save/npy/keras48_1_train_y.npy')  # (20,) 
x_test = np.load('keras/save/npy/keras48_1_test_x.npy')  # (20, 200, 200, 3)    
y_test = np.load('keras/save/npy/keras48_1_test_y.npy')  # (20,) 

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)  

#2. 모델링
model = Sequential()
model.add(Conv2D(32, (2, 2), padding='same', input_shape=(200, 200, 3), activation='relu'))
model.add(MaxPool2D(2))                                                    
model.add(Conv2D(16, (4, 4), activation='relu'))                      
model.add(Flatten())
model.add(Dense(36, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', patience=50, mode='max', verbose=1,  restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[es])    

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=10)
print('[loss] : ', loss[0])        
print('[acc] : ', loss[1])         

'''
[loss] :  1.0926132202148438
[acc] :  0.5
'''