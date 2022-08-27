import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,  #스케일링
    horizontal_flip=True,  # 좌우반전
    vertical_flip=True,  # 상하반전
    width_shift_range=0.1,  # 좌우이동
    height_shift_range=0.1,  # 상하이동
    rotation_range=5,  # 회전
    zoom_range=1.2,  # 확대
    shear_range=0.7,  # 기울기
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(
    rescale=1./255  # 평가데이터는 원본으로 사용해야 한다. -> 데이터 증폭X
) 

xy_train = train_datagen.flow_from_directory(
    'dacon/seoul_landmark/data/train/',
    target_size=(150, 150), 
    batch_size=5,
    class_mode='binary',
    shuffle=True,) 

xy_test = test_datagen.flow_from_directory(
    'dacon/seoul_landmark/data/test/',
    target_size=(150, 150),
    batch_size=5, 
    class_mode='binary',)  # 평가데이터에서 셔플은 필요없음

#2. 모델링
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

#3. Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit_generator(xy_train, epochs=30, steps_per_epoch=32,  # steps_per_epoch = 전체데이터/batch --> 160/5 = 32
                    validation_data=xy_test,
                    validation_steps=4,  # validation_steps = 전체데이터/batch --> 20/5 = 4
                    )
acc = hist.history['acc']
val_acc = hist.history['val_loss']
loss = hist.history['loss']
val_loss = hist.history['val_loss']



