from tensorflow.keras import datasets
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(    
    rescale=1./255,  # 스케일링
    horizontal_flip=True,  # 좌우반전
    # vertical_flip=True,  # 상하반전
    width_shift_range=0.1,            
    height_shift_range=0.1,   
    # rotation_range=5,               
    zoom_range=0.1,                 
    # shear_range=0.7,                    
    fill_mode='nearest'        
)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size)

# 랜덤으로 추출한 인덱스 카피
x_augmented = x_train[randidx].copy()  # (10, 28, 28)
y_augmented = y_train[randidx].copy()  # (10,)    

# 입력데이터 4차원 변환
x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)

# 카피 데이터 변환
x_augmented = train_datagen.flow(
    x_augmented, np.zeros(augment_size),  # 임의의 y_data
    batch_size=augment_size, shuffle=False
).next()[0]  # x_data만 추출

x_train = np.concatenate((x_train, x_augmented))  # 기존 x_data와 변환한 x_data 결합
y_train = np.concatenate((y_train, y_augmented))

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(2, 10, i+1)  # plt.subplot(row, column, index)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap='gray')
plt.show()   