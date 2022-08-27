from tensorflow.keras import datasets
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()
print(x_train.shape)  # (60000, 28, 28)  
print(x_train[0].shape)  # (28, 28)
print(x_train[0].reshape(28*28).shape)  # (784,) --> 1차원 벡터화 
print(np.tile(x_train[0].reshape(28*28), 100).shape)  # (78400,) 
print(np.tile(x_train[0].reshape(28*28), 100).reshape(-1, 28, 28, 1).shape)  # (100, 28, 28, 1)
'''
np.tile(A, repeat_shape) 형태이며, A 배열이 repeat_shape 형태로 반복되어 쌓인 형태
ex) np.tile(x_train[0].reshape(28*28), 100) -> x_train의 제일 첫번째 사진[0]을 (784,) 배열로 만들어 100번 반복, 여기서 100장의 사진이 완성
'''

train_datagen = ImageDataGenerator(    
    rescale=1./255,                    
    horizontal_flip=True, # 좌우반전
    # vertical_flip=True,  # 상하반전
    width_shift_range=0.1,  # 좌우이동
    height_shift_range=0.1,  # 상하이동
    # rotation_range=5,  # 회전
    zoom_range=0.1,  # 확대
    # shear_range=0.7,  # 기울기
    fill_mode='nearest'          
)

augment_size = 100
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1, 28, 28, 1),  # (100, 28, 28, 1) -> 흑백조 28*28 사이즈 이미지 100개
    np.zeros(augment_size),  # np.zeros(): augment_size만큼 0으로 채워진 array 생성
    batch_size=augment_size,  # 100개로 묶어줌   
    shuffle=False 
).next()  # next(): 반복 가능 객체의 다음 요소 반환

ic(len(x_data))
ic(type(x_data))  # <class 'tuple'>
ic(x_data[0].shape, x_data[1].shape)  # x_data: (100, 28, 28, 1) y_data: (100,)

plt.figure(figsize=(7, 7))
for i in range(49):
    plt.subplot(8,8,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')
plt.show()

