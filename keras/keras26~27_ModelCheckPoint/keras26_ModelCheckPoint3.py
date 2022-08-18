from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

#1 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# #2. 모델링 
model = Sequential()
model.add(Dense(40, input_dim=13))
model.add(Dense(30))
model.add(Dense(20, activation='relu'))
model.add(Dense(10))
model.add(Dense(1))

#3. 모델 컴파일, 훈련, 저장
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath='./_ModelCheckPoint/keras26_3_MCP.hdf5')
model.fit(x_train, y_train, epochs=50, batch_size=8, validation_split=0.25, callbacks=[es,mcp]) 
model.save("./_save/keras26_3_save_MCP.h5")

                       
#4. 평가, 예측                                               
print("======================= 1. 기본 출력 =========================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

print("======================= 2. load_model 출력 ======================")
model2 = load_model('./_save/keras26_3_save_MCP.h5')
loss2 = model2.evaluate(x_test, y_test)
print('loss2 : ', loss2)

y_predict2 = model2.predict(x_test)
r2 = r2_score(y_test, y_predict2) 
print('r2스코어 : ', r2)

print("====================== 3. mcp 출력 ============================")
model3 = load_model('./_ModelCheckPoint/keras26_3_MCP.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss3 : ', loss3)

y_predict3 = model3.predict(x_test)
r2 = r2_score(y_test, y_predict3) 
print('r2스코어 : ', r2)



'''
======================= 1. 기본 출력 =========================
[loss] :  36.11342239379883
[r2스코어] :  0.5679330459669675

======================= 2. load_model 출력 ======================
[loss2] :  36.11342239379883
[r2스코어] :  0.5679330459669675

====================== 3. mcp 출력 ============================
[loss3] :  30.61674690246582
[r2스코어] :  0.6336961556175824
'''