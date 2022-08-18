from tensorflow.keras import datasets
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

print(type(x_train),len(x_train),x_train.shape)

train_datagen = ImageDataGenerator(    
    rescale=1./255,                    
    horizontal_flip=True,               
    #vertical_flip=True,                                      
    width_shift_range=0.3,            
    height_shift_range=0.3,   
    #rotation_range=5,               
    zoom_range=0.3,                 
    #shear_range=0.7,                    
    fill_mode='nearest'        
)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0],size=augment_size)
# x_train은 60000,28,28 이중에 0번째를 가져왔으므로 60000 (0~59999)중에서 40000개를 랜덤하게 중복없이 뽑겠다는 의미.
# 이게 뭔 뜻이냐. 60000장중에 랜덤하게 40000장을 뽑아서 이 40000장을 변환하겠다. 
#print(x_train.shape[0])                     # 60000
#print(randidx)                              # 랜덤한 수가 40000개 들어있다.
#print(np.min(randidx), np.max(randidx))     # 0 59999

x_augmented = x_train[randidx].copy()      # .copy는 메모리에 한번 더 실어줘서 안정성을 주기위해 함.
y_augmented = y_train[randidx].copy()      # x,y 둘다 40000개이며 (40000,28,28) (40000,)
# 중요** 데이터셋 합칠때 merge와 concatenate 

x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(20):
    plt.subplot(8,8,i+1)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap='gray')
plt.show()  

x_augmented = train_datagen.flow(
    x_augmented, np.zeros(augment_size),
    batch_size=augment_size, shuffle=False
).next()[0]                                     
# 이미지데이터제너레이터해서 x값과 y값에 각각 값이 들어가는데 np.zeros로 일단 0값으로 다 들어가는데 .next()하고 [0]을 붙여줘서 
# x값만 가져와서 40000번 반복해서 여기서 y값은 drop된다. 

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(20):
    plt.subplot(8,8,i+1)
    plt.axis('off')
    plt.imshow(x_augmented[i], cmap='gray')
plt.show()   

#print(x_augmented)
#print(x_augmented.shape)

x_train = np.concatenate((x_train, x_augmented))   # concatenate 괄호 2개 씀 (()) 왜 why..? axis 때문?
y_train = np.concatenate((y_train, y_augmented))
#print(x_train)
#print(x_train.shape, y_train.shape)

#print(x_train[10000])
#print(y_train[10000])

# print(y_train[70000])
# print(y_train[80000])
# print(y_train[90000])
# print(y_train[99999])

print(type(x_train),len(x_train))
