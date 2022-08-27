import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
import time
import warnings
warnings.filterwarnings('ignore')
from icecream import ic

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
    validation_split=0.3,
    fill_mode= 'nearest')

train_generator = train_datagen.flow_from_directory(
    'keras/data/images/horse_or_human/train/',
    target_size=(150, 150),
    batch_size=719,
    class_mode='binary',
    subset='training'  # set as training data
)   # Found 719 images belonging to 2 classes.

validation_generator = train_datagen.flow_from_directory(
    'keras/data/images/horse_or_human/train/',  # same directory as training data
    target_size=(150, 150),
    batch_size=308,
    class_mode='binary',
    subset='validation'  # set as validation data
)   # Found 308 images belonging to 2 classes.

print(train_generator[0][0].shape)  # x_data (719, 150, 150, 3)
print(validation_generator[0][0].shape) # x_val (308, 150, 150, 3)

# 증폭 데이터 생성
augment_size = 500
randidx = np.random.randint(719, size=augment_size)  # 0~719 중 500개의 랜덤 인덱스 추출

x_augmented = train_generator[0][0][randidx].copy()   
y_augmented = train_generator[0][1][randidx].copy() 

# 카피 데이터 증폭
augmented_data = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=500, 
    shuffle=False
)

# 기존 데이터와 결합
x = np.concatenate((train_generator[0][0], augmented_data[0][0]))  
y = np.concatenate((train_generator[0][1], augmented_data[0][1]))

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
end = time.time() - start
print("걸린 시간 : ", round(end, 2))

#4. 평가, 예측 
loss = model.evaluate(validation_generator)  # x,y val_data
ic(loss[0], loss[1])

'''
[loss]: 0.4918624460697174, [acc]: 0.8279221057891846
'''