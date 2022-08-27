from tensorflow.keras import datasets
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


#1. 데이터
(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()
ic(type(x_train), len(x_train), x_train.shape) 
'''
type(x_train): <class 'numpy.ndarray'>
len(x_train): 60000
x_train.shape: (60000, 28, 28)
'''
train_datagen = ImageDataGenerator(    
    rescale=1./255,  # 스케일링
    horizontal_flip=True,  # 좌우반전
    #vertical_flip=True,  # 상하반전
    width_shift_range=0.3,  # 좌우이동
    height_shift_range=0.3,  # 상하이동
    #rotation_range=5,  # 회전
    zoom_range=0.3,  # 확대
    #shear_range=0.7,  # 기울기
    fill_mode='nearest'        
)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augment_size)  # np.random.randint(n, size) -> n범위에서 size만큼의 임의의 정수를 출력
# x_train.shape = (60000, 28, 28), x_train.shape[0] = 60000 --> 60000개 내에서 40000개의 임의의 정수 출력
ic(randidx) 
'''
[49754, 33291, 54091, ..., 25988, 45996, 39635]
'''

x_augmented = x_train[randidx].copy()  # (40000, 28, 28)
y_augmented = y_train[randidx].copy()  # (40000,)     

x_augmented = x_augmented.reshape(x_augmented.shape[0], x_augmented.shape[1], x_augmented.shape[2], 1)  #  4차원 변환
x_train = x_train.reshape(60000, 28, 28, 1)  # 4차원 변환 
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)  # 4차원 변환

plt.figure(figsize=(7, 7))
for i in range(20):
    plt.subplot(8, 8, i+1)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap='gray')
plt.show()  

x_augmented = train_datagen.flow(
    x_augmented, np.zeros(augment_size),
    batch_size=augment_size, shuffle=False
).next()[0]  # next()[0] -> x값만

plt.figure(figsize=(7,7))
for i in range(20):
    plt.subplot(8, 8, i+1)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap='gray')
plt.show()   

x_train = np.concatenate((x_train, x_augmented))  # type(x_train): <class 'numpy.ndarray'>, len(x_train): 100000
y_train = np.concatenate((y_train, y_augmented))
'''
np.concatenate((a1a1a_1,a2a2a_2,...), axis=0)
-a1a1a_1,a2a2a_2,... : ndarray이며 반드시 같은 shape여야 한다.
-axis : 2차원일 때 0=행, 1=열, 3차원일 때 0=깊이(차원), 1=행, 2=열 --> default 0

축에 따라서 수직, 수평, 깊이로 합칠 수 있고 axis로 설정하고 각 축마다 대체 메소드가 있다.
데이터를 쌓는 자료구조인 stack을 따서 만든 것으로 예상되는데,
수직(axis=0)일때는 np.vstack, 수평(axis=1)일때는 np.hstack, 깊이(depth)는 np.dstack으로 할 수 있다.
'''

ic(type(x_train), len(x_train))
