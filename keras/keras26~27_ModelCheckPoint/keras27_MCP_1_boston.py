from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
import numpy as np
import time

#1.데이터 로드 및 정제
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49)    

#scaler = MinMaxScaler()   
scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)    

#2. 모델구성,모델링
input1 = Input(shape=(13))
dense1 = Dense(50)(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(15, activation='relu')(dense2)
dense4 = Dense(8, activation='relu')(dense3)
dense5 = Dense(5)(dense4)
output1 = Dense(1)(dense5)
model = Model(inputs=input1, outputs=output1)


ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

#3. 모델 컴파일, 훈련, 저장
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_1_boston{krtime}_MCP.hdf5')
model.fit(x_train, y_train, epochs=10000, batch_size=10, validation_split=0.111111, callbacks=[es, mcp])
model.save(f"./_save/keras26_1_save_boston{krtime}.h5")

#4. 평가 예측
print("======================= 1. 기본 출력 =========================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

print("======================= 2. load_model 출력 ======================")
model2 = load_model(f"./_save/keras26_1_save_boston{krtime}.h5")
loss2 = model2.evaluate(x_test, y_test)
print('loss2 : ', loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2) 
print('r2스코어 : ', r2)

print("====================== 3. mcp 출력 ============================")
model3 = load_model(f'./_ModelCheckPoint/keras26_1_boston{krtime}_MCP.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss3 : ', loss3)
y_predict3 = model3.predict(x_test)
r2 = r2_score(y_test, y_predict3) 
print('r2스코어 : ', r2)


'''
======================= 1. 기본 출력 =========================
[loss] :  14.98935317993164
[r2스코어] :  0.8486742103572854

======================= 2. load_model 출력 ======================
[loss2] :  14.98935317993164
[r2스코어] :  0.8486742103572854

====================== 3. mcp 출력 ============================
[loss3] :  14.98935317993164
[r2스코어] :  0.8486742103572854
'''