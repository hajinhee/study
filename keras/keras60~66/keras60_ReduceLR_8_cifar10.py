from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np,time
from tensorflow.keras.datasets import cifar10 
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(50000,32,32,3)/255.
x_test = x_test.reshape(10000,32,32,3)/255.

#2. 모델링
model = Sequential()
model.add(Conv2D(10,kernel_size=(3,3),strides=1,padding='valid', input_shape=(32,32,3), activation='relu')) # 30,30,10
model.add(MaxPooling2D(2,2))                                                                                # 15,15,10
model.add(Conv2D(10,kernel_size=(2,2), strides=1, padding='same', activation='relu'))                       # 15,15,10
model.add(MaxPooling2D(3,3))                                                                                #  5,5,10
model.add(Conv2D(10,(2,2), activation='relu'))                                                              #  4,4,10
model.add(MaxPooling2D(2,2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
learning_rate = 0.001           
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['acc']) 

es = EarlyStopping(monitor="val_acc", patience=15, mode='max',verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, mode='max',verbose=1,min_lr=0.0001,factor=0.5) 

start = time.time()
model.fit(x_train,y_train,epochs=1000, batch_size=50,validation_split=0.2, callbacks=[reduce_lr,es])
end = time.time() - start

#4. 평가
loss , acc = model.evaluate(x_test,y_test)

print(f"lr : {learning_rate}, loss : {round(loss,4)}, acc : {round(acc,4)}, 걸린시간 : {round(end,4)}초")

# 기존 acc 0.65 # lr : 0.001, loss : 1.2538, acc : 0.5445, 걸린시간 : 657.1859초