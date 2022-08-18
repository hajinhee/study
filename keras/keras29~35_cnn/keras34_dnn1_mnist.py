from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time, numpy as np


#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape)  (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)    (10000, 28, 28) (10000,)

x_train = x_train.reshape(len(x_train),-1)  # (60000, 784)
x_test = x_test.reshape(len(x_test),-1)  # (10000, 784)

print(np.unique(y_train))  # [0 1 2 3 4 5 6 7 8 9]

scaler = MinMaxScaler()  # 스케일러는 숫자형태의 2차원 데이터만 입력이 가능하다
x_train = scaler.fit_transform(x_train)     
x_test = scaler.transform(x_test)

#2.모델링
input1 = Input(shape=(784))
dense1 = Dense(100)(input1)
dense6 = Dropout(0.2)(dense1)
dense2 = Dense(80)(dense6)
dense3 = Dense(60,activation='relu')(dense2)
dense7 = Dropout(0.4)(dense3)
dense4 = Dense(40,activation='relu')(dense7)
dense5 = Dense(20)(dense4)
output1 = Dense(10,activation='softmax')(dense5)  # 'softmax' -> 다중분류
model = Model(inputs=input1, outputs=output1)

#3.컴파일,훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['acc'])  # 'sparse_categorical_crossentropy' -> 다중분류 손실함수
'''
[categorical_crossentropy]: 'one-hot encoding' 클래스 다중 분류 손실함수
[sparse_categorical_crossentropy]: 'integer type' 클래스 다중 분류 손실함수
'''
es = EarlyStopping(monitor='val_acc', patience=50, mode='max', verbose=1, baseline=None, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras34_1_mnist{fn}_MCP.hdf5')
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, callbacks=[es])
end = time.time()
# model.save(f"./_save/keras34_1_save_mnist111.h5")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('걸린시간 : ', np.round(end-start, 4))
print('accuracy : ', np.round(loss[1], 4))

'''
[loss] :  0.12382596731185913
[걸린시간] :  107.5059
[accuracy] :  0.9646
'''
