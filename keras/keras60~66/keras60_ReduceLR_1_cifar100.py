import numpy as np,time,warnings
from tensorflow.keras.datasets import mnist, cifar100
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
# warnings.filterwarnings('ignore')

#1. 데이터
(x_train,y_train), (x_test,y_test) = cifar100.load_data()

x_train = x_train.reshape(50000,32,32,3)/255.
x_test = x_test.reshape(10000,32,32,3)/255.

#2. 모델
model = Sequential()

model.add(Conv2D(128,(2,2),padding='valid',activation='relu', input_shape=(32,32,3)))  
model.add(Dropout(0.2))
model.add(Conv2D(128,(2,2),padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64,(2,2), activation='relu'))
model.add(MaxPool2D())

model.add(Flatten())                  # Flatten은 쫙 펴서 데이터를 넘겨준다.
# model.add(GlobalAveragePooling2D())     # Global은 10개의 평균값을 넘겨준다?
model.add(Dense(100, activation='softmax'))

#3. 컴파일, 훈련
learning_rate = 0.001           # 초기 lr이 되게 중요하다 
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,metrics=['acc'])

es = EarlyStopping(monitor="val_loss", patience=15, mode='min',verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto',verbose=1,min_lr=0.0001,factor=0.5)   
# 값이 갱신이 안되는순간 LR을 0.5배만큼 ( * 0.5) 감소시키겠다 
# 모델이 돌아가다가 값이 5번 이내에 갱신이안되면 lr감소. 그리고 다시 5번 보다가 안되면 lr감소. 그래도 안되면 early스탑걸려서 멈춰지기때문에
# 둘 사이의 값 조정이 필요

start = time.time()
model.fit(x_train,y_train,epochs=1000, batch_size=64,validation_split=0.2, callbacks=[reduce_lr,es])
end = time.time() - start

loss, acc = model.evaluate(x_test,y_test,batch_size=32)

print(f"lr : {learning_rate}, loss : {round(loss,4)}, acc : {round(acc,4)}, 걸린시간 : {round(end,4)}초")

# GlobalAveragePooling
# lr : 0.001, loss : 2.2437, acc : 0.4225, 걸린시간 : 935.174초

# Flatten 
# lr : 0.001, loss : 2.5074, acc : 0.3822, 걸린시간 : 184.5592초