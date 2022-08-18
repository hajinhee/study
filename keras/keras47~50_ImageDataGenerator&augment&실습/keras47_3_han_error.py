import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. DATA
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
)

test_datagen = ImageDataGenerator(
    rescale=1./255
) #평가는 증폭해서는 안된다 


#이미지 폴더 정의 # D:\_data\image\brain
xy_train = train_datagen.flow_from_directory(
    '../_data/image/brain/train/',
    target_size=(150, 150), #사이즈는 지정된대로 바꿀수있다
    batch_size=5,
    class_mode='binary',
    shuffle=True,) #Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    '../_data/image/brain/test/',
    target_size=(150, 150),
    batch_size=5, #짜투리 남는것도 한배치로 돈다
    class_mode='binary',) #셔플은 필요없음
    #Found 120 images belonging to 2 classes.

#print(xy_train) #<tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000015005914F70>
#print(xy_train[0]) #첫번째 배치가 보임  dtype=float32), array([0., 0., 1., 1., 1.], dtype=float32)) 배치사이즈5를 줬기때문에 5개가 나옴
#print(xy_train[31]) #마지막 배치
#print(xy_train[32]) #ValueError: Asked to retrieve element 32, but the Sequence has length 32
#32개 밖에 없으므로 33번째를 호출하면 에러 // 0~5까지 배치사이즈를 나눴기 때문에

#print(xy_train[0][0]) # X 첫번째 배치의 첫X
#print(xy_train[0][1]) # Y
#print(xy_train[0][2]) # IndexError: tuple index out of range

#print(xy_train[0][0].shape) #(5, 100, 100, 3) #디폴트 채널은 3, 컬러다
#print(xy_train[0][1].shape) #(5,)
#print(type(xy_train))       #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
#print(type(xy_train[0]))    #<class 'tuple'>
#print(type(xy_train[0][0])) #<class 'numpy.ndarray'>


#2. MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

#3. Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# model.fit(xy_train[0][0], xy_train[0][1])
hist = model.fit_generator(xy_train, epochs=30, steps_per_epoch=32, #전체데이터/batch = 160/5 =32
                    validation_data=xy_test,
                    validation_steps=4,
                    )
acc = hist.history['acc']
val_acc = hist.history['val_loss']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#그래프 그려보세요

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('acc : ', acc[-1])
print('val_acc : ', val_acc[-1])

