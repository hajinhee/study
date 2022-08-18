from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from pandas import get_dummies
from tensorflow.python.keras.callbacks import ModelCheckpoint
import time


#1.데이터 로드 및 정제
datasets = load_wine()
x = datasets.data
y = datasets.target
y = get_dummies(y)  # 원핫인코딩 -> 분류모델
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49) 
 
scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)   
x_test = scaler.transform(x_test)    


#2. 모델링
input1 = Input(shape=(13))
dense1 = Dense(120)(input1)
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(80)(dense2)
dense4 = Dense(60, activation='relu')(dense3)
dense5 = Dense(40)(dense4)
dense6 = Dense(20)(dense5)
output1 = Dense(3, activation='softmax')(dense6)  # 'softmax' -> 다중분류
model = Model(inputs=input1, outputs=output1)

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

#3. 컴파일, 훈련, 저장
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 'categorical_crossentropy' -> 다중분류
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,baseline=None, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_5_wine{krtime}_MCP.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=1, validation_split=0.1111111111, callbacks=[es,mcp])
model.save(f"./_save/keras26_5_save_wine{krtime}.h5")

#4. 평가 예측
print("======================= 1. 기본 출력 =========================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

print("======================= 2. load_model 출력 ======================")
model2 = load_model(f"./_save/keras26_5_save_wine{krtime}.h5")
loss2 = model2.evaluate(x_test, y_test)
print('loss2 : ', loss2[0])
print('accuracy2 : ', loss2[1])

print("====================== 3. mcp 출력 ============================")
model3 = load_model(f'./_ModelCheckPoint/keras26_5_wine{krtime}_MCP.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss3 : ', loss3[0])
print('accuracy3 : ', loss3[1])


'''
======================= 1. 기본 출력 =========================
[loss] :  0.14911648631095886
[accuracy] :  0.9444444179534912
======================= 2. load_model 출력 ======================
[loss2] :  0.14911648631095886
[accuracy2] :  0.9444444179534912
====================== 3. mcp 출력 ============================
[loss3] :  0.14911648631095886
[accuracy3] :  0.9444444179534912
''' 