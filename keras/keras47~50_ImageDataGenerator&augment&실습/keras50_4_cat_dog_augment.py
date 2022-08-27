import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
import time
from sklearn.metrics import accuracy_score 
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
train_datagen = ImageDataGenerator(  
    rescale=1./255, 
    horizontal_flip = True,
    # vertical_flip = True,
    width_shift_range= 0.1,
    height_shift_range= 0.1,
    # rotation_range= 5,
    zoom_range = 0.1,
    # shear_range = 0.7,
    fill_mode= 'nearest')

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

xy_train = train_datagen.flow_from_directory(         
    'keras/data/images/cat_or_dog/train/',
    target_size=(150, 150),                         
    batch_size=8001,
    class_mode='binary',
    shuffle=True,
)   # Found 402 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    'keras/data/images/cat_or_dog/test/',
    target_size=(150, 150),
    batch_size=2020, 
    class_mode='binary',
)   # Found 202 images belonging to 2 classes.

print(xy_train[0][0].shape)  # x_data (402, 150, 150, 3)
print(xy_train[0][1].shape)  # y_data (402,)
print(xy_test[0][0].shape)  # x_test (202, 150, 150, 3)
print(xy_test[0][1].shape)  # y_test (202,)

# 증폭 데이터 생성
augment_size = 200
randidx = np.random.randint(402, size=augment_size)  # 0~402 안에서 200개 랜덤 숫자 추출

x_augmented = xy_train[0][0][randidx].copy()  # x_data copy
y_augmented = xy_train[0][1][randidx].copy()  # y_data copy

# copy data 증폭
augmented_data = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=200, 
    shuffle = False
    )

# 기존 데이터와 결합
x = np.concatenate((xy_train[0][0], augmented_data[0][0]))  
y = np.concatenate((xy_train[0][1], augmented_data[0][1]))

#2. 모델링
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(150, 150, 3)))
model.add(Conv2D(64, (2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])  

start = time.time()
model.fit(x, y, epochs=10, steps_per_epoch=1000) 
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

#4. 평가, 예측 
loss = model.evaluate(xy_test)
print('[loss] : ', loss[0], '[acc]: ', loss[1])

y_predict = model.predict(xy_test[0][0])
y_predict = np.argmax(y_predict, axis=1) 
accuracy = accuracy_score(xy_test[0][1], y_predict)
print('[accuracy]: ', accuracy)

'''
[loss] :  1.2466084957122803 [acc]:  0.5247524976730347
[accuracy]:  0.5
'''