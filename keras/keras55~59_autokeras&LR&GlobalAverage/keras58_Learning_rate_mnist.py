import numpy as np,time
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#1. load data 
(x_train,y_train), (x_test,y_test) = mnist.load_data()  # 분류

# reshape, normalize
x_train = x_train.reshape(60000, 28, 28, 1)/255.  # (60000, 28, 28) -> (60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)/255.

#2. modeling
model = Sequential()
model.add(Conv2D(128, (2,2), padding='valid', activation='relu', input_shape=(28, 28, 1)))  
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())       
model.add(Dense(10, activation='softmax'))

#3. compile, train
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)

start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[es])
end = time.time() - start

#4. evaluate, predict
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('[loss]: ', loss, '[accuracy]: ', acc)
print(f'[lr]: {learning_rate}, [loss]: {round(loss,4)}, [accuracy]: {round(acc,4)}, [time]: {round(end,4)}초')


'''
epochs 10
[lr]: 0.01, [loss]: 0.3798, [accuracy]: 0.8728, [time]: 96.3905초
[lr]: 0.001, [loss]: 0.065, [accuracy]: 0.9798, [time]: 93.3999초
[lr]: 0.0001, [loss]: 0.0552, [accuracy]: 0.9822, [time]: 97.9721초
'''