import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image
from icecream import ic

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale = 1./255,              
    horizontal_flip = True,        
    vertical_flip= True,           
    width_shift_range = 0.1,       
    height_shift_range= 0.1,       
    rotation_range= 5,
    zoom_range = 1.2,              
    shear_range=0.7,
    fill_mode = 'nearest',
    validation_split=0.3          
    )                   # set validation split

batch = 5
train_generator = train_datagen.flow_from_directory(
    'keras/data/images/rps/',
    target_size=(100, 100),
    batch_size=batch,
    class_mode='categorical',
    shuffle=True,
    subset='training')  # set as training split

validation_generator = train_datagen.flow_from_directory(
    'keras/data/images/rps/',
    target_size=(100, 100),
    batch_size=batch,
    class_mode='categorical',
    subset='validation')  # set as validation split

print(train_generator[0][0].shape)  # x_data (5, 100, 100, 3)
print(train_generator[0][1].shape)  # y_data (5, 3)
print(validation_generator[0][0].shape)  # x_test (5, 100, 100, 3)
print(validation_generator[0][1].shape)  # y_test (5, 3)

# np.save('keras/save/npy/keras48_3_train_x.npy', arr = train_generator[0][0]) 
# np.save('keras/save/npy/keras48_3_train_y.npy', arr = train_generator[0][1])
# np.save('keras/save/npy/keras48_3_test_x.npy', arr = validation_generator[0][0])
# np.save('keras/save/npy/keras48_3_test_y.npy', arr = validation_generator[0][1])

#2. 모델링
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(100, 100, 3)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(40, activation='relu')) 
model.add(Dense(30, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 모델 컴파일, 훈련, 저장
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics=['acc'])
hist = model.fit_generator(train_generator, epochs=30, steps_per_epoch=len(train_generator), 
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator))
# model.save('./_save/keras48_3_save.h5')

#4. 평가, 확인
loss, acc = model.evaluate(validation_generator)   
ic(loss, acc)

'''
[loss]:  0.7148408889770508 [acc]:  0.7297710180282593
'''

# 샘플 케이스 경로지정
sample_image = 'keras/data/images/rps_sample.jpg'

# 샘플 케이스 확인
image_ = plt.imread(str(sample_image))
plt.title("Test Case")
plt.imshow(image_)
plt.axis('Off')
# plt.show()

# 샘플 이미지 예측
image_ = keras_image.load_img(str(sample_image), target_size=(100, 100))  # 이미지 로드 <class 'PIL.Image.Image'>
x = keras_image.img_to_array(image_)  # Image를 numpy.array로 변환 --> (100, 100, 3) 3차원
x = np.expand_dims(x, axis=0) # 차원 추가 --> (1, 100, 100, 3) 4차원
x /= 255.  # 픽셀값을 0~1 사이로 정규화
x = np.vstack([x])  # vstack: 배열을 수직(행방향)으로 쌓는다. hstack: 배열을 수평(열방향)으로 쌓는다.
classes = model.predict(x, batch_size=1)  #  classes: [[0.21934389, 0.37464112, 0.40601498]]
ic(classes)
y_predict = np.argmax(classes)  # 가장 큰 원소의 인덱스 반환 --> y_predict:  0

ic(validation_generator.class_indices)  # 클래스 이름과 클래스 색인 간 매핑을 담은 딕셔너리
                                        # class_indices: {'paper': 0, 'rock': 1, 'scissors': 2}

print(f'{classes[0][0]*100} 의 확률로 [보]입니다.' if y_predict==0 
        else f'{classes[0][1]*100} 의 확률로 [바위]입니다.' if y_predict==1 
        else f'{classes[0][2]*100} 의 확률로 [가위]입니다.')

'''
39.96744155883789 의 확률로
 → '보'입니다.
'''