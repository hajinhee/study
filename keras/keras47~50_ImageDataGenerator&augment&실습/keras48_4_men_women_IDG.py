import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from icecream import ic
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

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
    validation_split=0.3  # set validation split       
    )                   

train_generator = train_datagen.flow_from_directory(
    'keras/data/images/men_or_women/',
    target_size=(100, 100),
    batch_size=10,
    class_mode='binary',
    subset='training')  # set training data 

validation_generator = train_datagen.flow_from_directory(
    'keras/data/images/men_or_women/', 
    target_size=(100, 100),
    batch_size=10,
    class_mode='binary',
    subset='validation')  # set validation data 

# np.save('keras/save/npy/keras48_4_train_x.npy', arr = train_generator[0][0])  # x_train
# np.save('keras/save/npy/keras48_4_train_y.npy', arr = train_generator[0][1])  # y_train
# np.save('keras/save/npy/keras48_4_test_x.npy', arr = validation_generator[0][0])  # x_test
# np.save('keras/save/npy/keras48_4_test_y.npy', arr = validation_generator[0][1])  # y_test

#2. 모델
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(100, 100, 3)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu')) 
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련, 저장
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])
hist = model.fit_generator(train_generator, epochs=5, steps_per_epoch=len(train_generator), 
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator))
# model.save('./_save/keras48_4_save_menwomen.h5')

#4. 평가, 예측
loss = model.evaluate_generator(validation_generator, steps=5)
ic(loss[0], loss[1])
'''
loss[0]: 0.669445276260376, loss[1]: 0.6200000047683716
'''

# 샘플 케이스 확인
img_path = 'keras/data/images/sample.jpg'
image_ = plt.imread(str(img_path))
plt.title("Test Case")
plt.imshow(image_)
plt.axis('Off')
# plt.show()

# 샘플 케이스 예측 
image_ = image.load_img(str(img_path), target_size=(100, 100))  # image
x = image.img_to_array(image_)  # 3차원 배열
x = np.expand_dims(x, axis=0)  # 4차원 배열
x /=255.  # 0~1 픽셀 정규화
images = np.vstack([x])  # 배열 수직 결합
classes = model.predict(images, batch_size=1)
ic(classes)  # [[0.5207067]]
y_predict = classes[0][0]
# ic(y_predict)  # 0
# ic(validation_generator.class_indices)  #  {'men': 0, 'women': 1}
result = f'{round(100-y_predict*100, 2)}% 확률로 남자입니다' if y_predict<=0.5 else f'{round(y_predict*100, 2)}% 확률로 여자입니다'
ic(result)
'''
result: '52.07% 확률로 여자입니다'
'''