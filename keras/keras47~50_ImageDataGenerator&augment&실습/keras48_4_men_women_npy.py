import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from icecream import ic

#1. 데이터
x_train = np.load('keras/save/npy/keras48_4_train_x.npy')
y_train = np.load('keras/save/npy/keras48_4_train_y.npy')
x_test = np.load('keras/save/npy/keras48_4_test_x.npy')
y_test = np.load('keras/save/npy/keras48_4_test_y.npy')

#2. 모델링
model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(100, 100, 3)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu')) 
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1) 
hist = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[es])  

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
ic(loss[0], loss[1])  #  loss[0]: 1.8679794073104858, loss[1]: 0.699999988079071

# 샘플 케이스 경로지정
sample_image = 'keras/data/images/rps_sample.jpg'

# 샘플 케이스 확인
# image_ = plt.imread(str(sample_image))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

# 샘플 케이스 예측
image_ = image.load_img(str(sample_image), target_size=(100, 100))  # image load
x = image.img_to_array(image_)  # image -> numpy.array(3d)
x = np.expand_dims(x, axis=0)  # 3d -> 4d
x /=255.  # 픽셀 정규화
images = np.vstack([x])  # 배열 수직(행방향)으로 쌓기
classes = model.predict(images, batch_size=40)  # [[0.0012185]]
y_predict = classes[0][0]

result = f'{round(100-y_predict*100, 2)}% 확률로 남자입니다' if y_predict<=0.5 else f'{round(y_predict*100, 2)}% 확률로 여자입니다'
ic(result)
'''
result: '99.88% 확률로 남자입니다'
'''