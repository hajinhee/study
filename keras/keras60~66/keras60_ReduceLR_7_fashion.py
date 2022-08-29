from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np,time
from tensorflow.keras.datasets import fashion_mnist  
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from icecream import ic

#1. load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
ic(x_train.shape)  # (60000, 28, 28)

# reshape
x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1)

#2. modeling
model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2), input_shape=(28,28,1)))  
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. compile, train 
learning_rate = 0.01           
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
es = EarlyStopping(monitor='val_accuracy', patience=15, mode='max', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5, mode='max', verbose=1, min_lr=0.001, factor=0.5) 

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.2, callbacks=[reduce_lr, es])
end = time.time()-start

#4. evaluate
loss, acc = model.evaluate(x_test, y_test)
ic(learning_rate, round(loss, 4), round(acc, 4), f'걸린시간: {round(end, 4)}')


'''
learning_rate: 0.01
loss: 0.3853
acc: 0.8588
걸린시간: 120.905
'''