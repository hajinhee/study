import numpy as np,time
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


#1. 데이터
(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)/255.
x_test = x_test.reshape(10000,28,28,1)/255.

#2. 모델
model = Sequential()

model.add(Conv2D(128,(2,2),padding='valid',activation='relu', input_shape=(28,28,1)))  
model.add(Dropout(0.2))
model.add(Conv2D(128,(2,2),padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64,(2,2), activation='relu'))
model.add(MaxPool2D())

# model.add(Flatten())                  # Flatten은 쫙 펴서 데이터를 넘겨준다.
model.add(GlobalAveragePooling2D())     # Global은 10개의 평균값을 넘겨준다?
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['acc'])

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1, restore_best_weights=True)

start = time.time()
model.fit(x_train,y_train,epochs=10, batch_size=32,validation_split=0.2, callbacks=[es])
end = time.time() - start

loss, acc = model.evaluate(x_test,y_test,batch_size=32)

print(f"lr : {learning_rate}, loss : {round(loss,4)}, acc : {round(acc,4)}, 걸린시간 : {round(end,4)}초")

# epochs 10고정

# lr : 0.01, loss : 0.3798, acc : 0.8728, 걸린시간 : 96.3905초
# lr : 0.001, loss : 0.065, acc : 0.9798, 걸린시간 : 93.3999초
# lr : 0.0001, loss : 0.0552, acc : 0.9822, 걸린시간 : 97.9721초