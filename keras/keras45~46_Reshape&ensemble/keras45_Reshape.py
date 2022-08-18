from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout, Activation, MaxPooling2D, Reshape, Conv1D,LSTM
from tensorflow.keras.datasets import mnist

model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),strides=1 ,padding='same',input_shape=(28,28,1)))                                 
model.add(MaxPooling2D())     
model.add(Conv2D(5,(2,2), activation='relu'))               # 13,13,5
model.add(Conv2D(7,(2,2), activation='relu'))               # 12,12,7  
model.add(Conv2D(7,(2,2), activation='relu'))               # 11,11,7  
model.add(Conv2D(10,(2,2), activation='relu'))              # 10,10,10  
model.add(Flatten())                                        # n,1000
model.add(Reshape(target_shape=(100,10)))                   # 
model.add(Conv1D(5,2))
model.add(LSTM(15))
model.add(Dense(10,activation='softmax'))
# model.add(Dense(32))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Dense(5, activation='softmax'))
model.summary()
 



