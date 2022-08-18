from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation
import numpy as np,time
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1)

# enco = OneHotEncoder(sparse=False)

# y_train = enco.fit_transform(y_train.reshape(-1,1)) 
# y_test = to_categorical(y_test)

#2. 모델링
model = Sequential()
model.add(Conv2D(10,kernel_size=(3,3), input_shape=(28,28,1)))  
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
learning_rate = 0.001           
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['accuracy']) 

es = EarlyStopping(monitor="val_accuracy", patience=15, mode='max',verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5, mode='max',verbose=1,min_lr=0.0001,factor=0.5) 

start = time.time()
model.fit(x_train,y_train,epochs=1000, batch_size=1000,validation_split=0.2, callbacks=[reduce_lr,es])
end = time.time() - start

#4. 평가
loss , acc = model.evaluate(x_test,y_test)

print(f"lr : {learning_rate}, loss : {round(loss,4)}, acc : {round(acc,4)}, 걸린시간 : {round(end,4)}초")

# 기존 0.98 -> lr : 0.001, loss : 0.0545, acc : 0.9814, 걸린시간 : 119.0547초