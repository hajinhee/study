import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

#1. 데이터 로드 및 전처리
train_datagen = ImageDataGenerator(    
    rescale=1./255,
    horizontal_flip=True,  # 좌우반전
    vertical_flip=True,  # 상하반전
    width_shift_range=0.1,  # 좌우이동
    height_shift_range=0.1,  # 상하이동
    rotation_range=5,  # 회전
    zoom_range=1.2,  # 부분확대
    shear_range=0.7,  # 기울기
    fill_mode='nearest'          
)

test_datagen = ImageDataGenerator(
    rescale=1./255  # 테스트 데이터는 원본을 사용해야되기 때문에 스케일링만 한다. 
)

xy_train = train_datagen.flow_from_directory(      
    'dacon/seoul_landmark/data/train/',
    target_size = (150, 150),                                                                       
    batch_size=5,                                   
    class_mode='binary',                         
    shuffle=True,    
)   

xy_test = test_datagen.flow_from_directory(         
    'dacon/seoul_landmark/data/test/',
    target_size=(150, 150),
    batch_size=5,
    class_mode='binary',                            
)  

#2. 모델링
model = Sequential()
model.add(Conv2D(32, (2, 2), padding='same', input_shape=(150, 150, 3), activation='relu'))
model.add(MaxPool2D(2))                    
model.add(Conv2D(16, (4, 4), activation='relu')) 
model.add(MaxPool2D(4))
model.add(Flatten())
model.add(Dense(36, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', patience=50, mode='max', verbose=1, restore_best_weights=True)
hist = model.fit_generator(xy_train, epochs=100, steps_per_epoch=32, callbacks=[es], validation_data=xy_test, validation_steps=4)

'''
첫번째 인자 : 훈련데이터셋을 제공할 제네레이터를 지정합니다. 본 예제에서는 앞서 생성한 xy_train으로 지정합니다.
epochs : 전체 훈련 데이터셋에 대해 학습 반복 횟수를 지정합니다. 100번을 반복적으로 학습시켜 보겠습니다.
steps_per_epoch : 한 epoch에 사용한 스텝 수를 지정합니다. 총 160개의 훈련 샘플이 있고 배치사이즈가 5이므로 32 스텝으로 지정합니다. (트레인 데이터 수 / 배치사이즈) = len(xy_train)
validation_data : 검증데이터셋을 제공할 제네레이터를 지정합니다. 본 예제에서는 앞서 생성한 xy_test으로 지정합니다.
validation_steps : 한 epoch 종료 시 마다 검증할 때 사용되는 검증 스텝 수를 지정합니다. 홍 120개의 검증 샘플이 있고 배치사이즈가 5이므로 24 스텝으로 지정합니다. (테스트 데이터 수 / 배치사이즈) = len(xy_test)
'''

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-51])
print('val_loss : ', val_loss[-51])
print('acc : ', acc[-51])
print('val_acc : ', val_acc[-51])

# 시각화
plt.plot(acc, color='red', marker='.',label='acc') 
plt.plot(val_acc, color='purple', marker='.',label='val_acc') 
plt.plot(loss, color='green', marker='.',label='loss') 
plt.plot(val_loss, color='blue', marker='.',label='val_loss') 
plt.ylabel('values')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()




