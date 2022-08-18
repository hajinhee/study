from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time

#1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# print(x_train.shape, y_train.shape) (60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape) (10000, 28, 28) (10000,)

x_train = x_train.reshape(len(x_train), -1)  # (60000, 784)
x_test = x_test.reshape(len(x_test), -1)  # (10000, 784)

y_train = to_categorical(y_train)  # 원핫인코딩
y_test = to_categorical(y_test)

scaler =MinMaxScaler()  
x_train = scaler.fit_transform(x_train)    
x_test = scaler.transform(x_test)

#2.모델링
input1 = Input(shape=(784))
dense1 = Dense(100)(input1)
dense6 = Dropout(0.2)(dense1)
dense2 = Dense(80)(dense6)
dense3 = Dense(60, activation='relu')(dense2)
dense7 = Dropout(0.4)(dense3)
dense4 = Dense(40, activation='relu')(dense7)
dense5 = Dense(20)(dense4)
output1 = Dense(10, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")
acc = '{accuracy:.4f}'
fn = "".join([krtime,acc])

#3.컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])  # 'categorical_crossentropy' -> 다중분류
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras34_2_fashion_mnist{fn}_MCP.hdf5')
model.fit(x_train, y_train,epochs=100, batch_size=100, validation_split=0.1111111111, callbacks=[es])
model.save(f"./_save/keras34_2_fashion_mnist{krtime}.h5")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
[loss] :  0.37753260135650635
[accuracy] :  0.8658999800682068
'''
