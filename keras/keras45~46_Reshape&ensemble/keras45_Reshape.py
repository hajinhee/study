from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout, Activation, MaxPooling2D, Reshape, Conv1D,LSTM
from tensorflow.keras.datasets import mnist

model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), strides=1, padding='same', input_shape=(28, 28, 1)))  # Conv2D -> 4차원 데이터                            
model.add(MaxPooling2D())     
model.add(Conv2D(5, (2, 2), activation='relu')) 
model.add(Conv2D(7, (2, 2), activation='relu'))
model.add(Conv2D(7, (2, 2), activation='relu')) 
model.add(Conv2D(10, (2, 2), activation='relu')) 
model.add(Flatten())                          
model.add(Reshape(target_shape=(100, 10)))  
model.add(Conv1D(5, 2))
model.add(LSTM(15))
model.add(Dense(10,activation='softmax'))
model.summary()
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 10)        50

 max_pooling2d (MaxPooling2D  (None, 14, 14, 10)       0
 )

 conv2d_1 (Conv2D)           (None, 13, 13, 5)         205

 conv2d_2 (Conv2D)           (None, 12, 12, 7)         147

 conv2d_3 (Conv2D)           (None, 11, 11, 7)         203

 conv2d_4 (Conv2D)           (None, 10, 10, 10)        290

 flatten (Flatten)           (None, 1000)              0

 reshape (Reshape)           (None, 100, 10)           0

 conv1d (Conv1D)             (None, 99, 5)             105

 lstm (LSTM)                 (None, 15)                1260

 dense (Dense)               (None, 10)                160

=================================================================
Total params: 2,420
Trainable params: 2,420
Non-trainable params: 0
_________________________________________________________________
'''



