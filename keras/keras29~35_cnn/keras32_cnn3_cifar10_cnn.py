from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np
from tensorflow.keras.datasets import cifar10 # 교육용데이터 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from pandas import get_dummies
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import ssl      

#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
# print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)

# plt.imshow(x_train[2],'gray')
# plt.show()
# print(np.unique(y_train, return_counts=True))  # 0~9까지 각각 5000개씩 10개의 label값.

enco = OneHotEncoder(sparse=False)
y_train = enco.fit_transform(y_train)   
y_test = to_categorical(y_test)


# 4차원 데이터의 스케일러 적용하는 방법
scaler = MinMaxScaler()  
print(x_train.shape)  # (10000, 32, 32, 3)
print(x_train.ndim)   # 4차원

'''
x_train = x_train.reshape(50000,-1)  # (50000, 32, 32, 3) --> (50000, 3072) 2차원
x_test = x_test.reshape(10000,-1)    # 스케일러는 2차원에서만 적용되어 데이터를 정제해주기 때문에 2차원 형태로 변환해줘야 한다.

x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32, 32, 3)  # 다시 원래 형태(4차원)로 돌려준다. 
x_test = x_test.reshape(10000, 32, 32, 3)

위의 일련의 작업들을 압축하면 아래와 같다.
'''
                               # 2차원 전환                   # 원핫인코딩 후 다시 4차원화
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test),-1)).reshape(x_test.shape)
# print(x_train[:2], x_train.shape)  
# print(x_test[:2], x_test.shape) 

#2. 모델링
model = Sequential()
model.add(Conv2D(10, kernel_size=(3, 3), strides=1, padding='valid', input_shape=(32, 32, 3), activation='relu')) 
model.add(MaxPooling2D(2, 2))                                                                               
model.add(Conv2D(10, kernel_size=(2, 2), strides=1, padding='same', activation='relu'))           
model.add(MaxPooling2D(3, 3))                                                                 
model.add(Conv2D(10, (2, 2), activation='relu'))                                                           
model.add(MaxPooling2D(2, 2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, baseline=None, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras32_cifar10_Minmax_MCP.hdf5')
model.fit(x_train, y_train, epochs=500, batch_size=1000, validation_split=0.2, callbacks=[es, mcp])
model.save(f"./_save/keras32_save_cifar10_Minmax.h5")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
loss :  1.246515154838562
accuracy :  0.5501000285148621
'''
