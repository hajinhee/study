from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from pandas import get_dummies
import time

#1.데이터 로드 및 정제
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

y = get_dummies(y)  # 분류모델
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49) 

#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
#x_train = scaler.fit_transform(x_train)   
#x_test = scaler.transform(x_test)    


#2. 모델링
input1 = Input(shape=(54))
dense1 = Dense(100, activation='relu')(input1)
dense2 = Dense(80)(dense1)
dense3 = Dense(60, activation='relu')(dense2)
dense4 = Dense(40)(dense3)
dense5 = Dense(20)(dense4)
output1 = Dense(7, activation='softmax')(dense5)  # 'softmax' -> 다중분류
model = Model(inputs=input1, outputs=output1)

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

#3. 컴파일, 훈련, 저장
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 'categorical_crossentropy' -> 다중분류
es = EarlyStopping(monitor='val_loss', patience=10, mode='min',verbose=1,baseline=None, restore_best_weights=False)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_6_fetch_covtype{krtime}_MCP.hdf5')
model.fit(x_train, y_train, epochs=10000, batch_size=100000, validation_split=0.11111111, callbacks=[es, mcp])
model.save(f"./_save/keras26_6_save_covtype{krtime}.h5")

#4. 평가 예측
print("======================= 1. 기본 출력 =========================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

print("======================= 2. load_model 출력 ======================")
model2 = load_model(f"./_save/keras26_6_save_covtype{krtime}.h5")
loss2 = model2.evaluate(x_test, y_test)
print('loss2 : ', loss2[0])
print('accuracy2 : ', loss2[1])

print("====================== 3. mcp 출력 ============================")
model3 = load_model(f'./_ModelCheckPoint/keras26_6_fetch_covtype{krtime}_MCP.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss3 : ', loss3[0])
print('accuracy3 : ', loss3[1])

'''
======================= 1. 기본 출력 =========================
[loss] :  5.233202934265137
[accuracy] :  0.3939967751502991
======================= 2. load_model 출력 ======================
[loss2] :  5.233202934265137
[accuracy2] :  0.3939967751502991
====================== 3. mcp 출력 ============================
[loss3] :  4.017263889312744
[accuracy3] :  0.5049567818641663
'''