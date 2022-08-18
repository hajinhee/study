from tensorflow.keras import datasets
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(    
    rescale=1./255,                    
    horizontal_flip=True,               
    #vertical_flip=True,                                      
    width_shift_range=0.1,            
    height_shift_range=0.1,   
    #rotation_range=5,               
    zoom_range=0.1,                 
    #shear_range=0.7,                    
    fill_mode='nearest'        
)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0],size=augment_size)

x_augmented = x_train[randidx].copy()     
y_augmented = y_train[randidx].copy()      

x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_augmented = train_datagen.flow(
    x_augmented, np.zeros(augment_size),
    batch_size=augment_size, shuffle=False
).next()[0]                                     

x_train = np.concatenate((x_train, x_augmented))   # concatenate 괄호 2개 씀 (()) 왜 why..? axis 때문?
y_train = np.concatenate((y_train, y_augmented))

# 점심숙제  x_augumented 10개와 변환되기전의 x_train 10개를 비교하는 이미지 출력할것 subplot(2,10,?) 사용

#print(x_augmented[randidx])

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.axis('off')
    plt.imshow(x_augmented[i],cmap='gray')
plt.show()   