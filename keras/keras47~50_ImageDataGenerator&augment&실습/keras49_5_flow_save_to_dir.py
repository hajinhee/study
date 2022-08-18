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

augment_size = 10
randidx = np.random.randint(x_train.shape[0],size=augment_size)

x_augmented = x_train[randidx].copy()     
y_augmented = y_train[randidx].copy()      

x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

import time

start_time = time.time()
print("시작!!")
x_augmented = train_datagen.flow(
              x_augmented, y_augmented,
              batch_size=augment_size, shuffle=False,
              save_to_dir="../_temp/"
            ).next()[0]                                     
end_time = time.time() - start_time

print('걸린시간 : ', round(end_time,3), '초')

x_train = np.concatenate((x_train, x_augmented))   
y_train = np.concatenate((y_train, y_augmented))


