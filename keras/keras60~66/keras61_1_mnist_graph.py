from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation
import numpy as np,time
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from icecream import ic
import matplotlib.pyplot as plt

#1. load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
ic(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)

# reshape
x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1)

#2. modeling
model = Sequential()
model.add(Conv2D(10,kernel_size=(3,3), input_shape=(28, 28, 1)))  
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
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc']) 
es = EarlyStopping(monitor='val_acc', patience=15, mode='max', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, mode='max', verbose=1, min_lr=0.0001, factor=0.5) 

start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=50, validation_split=0.2, callbacks=[reduce_lr, es])
end = time.time()-start

#4. evaluate
loss, acc = model.evaluate(x_test, y_test)
ic(loss, acc)
'''
loss: 0.16130343079566956, acc: 0.9516000151634216
'''
ic(learning_rate, round(loss, 4), round(acc, 4), f'걸린 시간: {round(end, 4)}')

'''
learning_rate: 0.01
loss: 0.1613
acc: 0.9516
걸린 시간: 159.2848
'''

####################### visualization ###########################
plt.figure(figsize=(9, 5))

#1 
plt.subplot(2, 1, 1)  # plt.subplot(nrows=2, ncols=1, index=1): 여러 개의 그래프, 축 공유 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

#2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])   # 위치 지정 안해주면 자동으로 빈 자리에 넣어준다.
plt.show()