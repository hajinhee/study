from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation
import numpy as np,time
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from icecream import ic
from tensorflow.keras.utils import to_categorical

#1. load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
ic(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)

# reshape
x_train = x_train.reshape(60000, 28, 28, 1)  # (60000, 28, 28) -> (10000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)  
ic(x_test.shape)
# check target
ic(np.unique(y_train, return_counts=True))

# one hot encoding -> sparse_categorical_crossentropy로 대체
# y_train = to_categorical(y_train)  # (60000, 10)
# y_test = to_categorical(y_test)  # (10000, 10)
# ic(y_train.shape, y_test.shape)

#2. modeling
model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3), input_shape=(28,28,1)))  
model.add(Conv2D(10, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), activation='relu'))
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
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5, mode='max', verbose=1, min_lr=0.0001, factor=0.5) 

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.2, callbacks=[reduce_lr, es])
end = time.time() - start

#4. evaluate
loss, acc = model.evaluate(x_test, y_test)
ic(learning_rate, round(loss, 4), round(acc, 4), f'걸린 시간: {round(end, 4)}')


'''
learning_rate: 0.01
loss: 0.0768
acc: 0.9766
걸린 시간: 144.859
'''