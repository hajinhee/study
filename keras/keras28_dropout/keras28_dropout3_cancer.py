from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
import numpy as np
import time

#1.데이터 로드 및 정제
datasets = load_boston()
x = datasets.data  # (506, 13)
y = datasets.target

print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49)    

# scaler = MinMaxScaler()   
scaler = StandardScaler()
#scaler = RobustScaler()
# scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)    


#2. 모델구성,모델링
model = Sequential()
model.add(Dense(30, input_dim=13))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=False, filepath=f'./_ModelCheckPoint/keras28_3_cancer{krtime}_MCP.hdf5')
model.fit(x_train, y_train, epochs=500, batch_size=3, validation_split=0.111111, callbacks=[es])
#model.save(f"./_save/keras28_3_save_cancer{krtime}.h5")

#4. 평가 예측
print("======================= 1. 기본 출력 =========================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

# print("======================= 2. load_model 출력 ======================")
# model2 = load_model(f"./_save/keras26_3_save_cancer{krtime}.h5")
# loss2 = model2.evaluate(x_test, y_test)
# print('loss2 : ', loss2)

# y_predict2 = model2.predict(x_test)
# r2 = r2_score(y_test, y_predict2) 
# print('r2스코어 : ', r2)

# print("====================== 3. mcp 출력 ============================")
# model3 = load_model(f'./_ModelCheckPoint/keras26_3_cancer{krtime}_MCP.hdf5')
# loss3 = model3.evaluate(x_test, y_test)
# print('loss3 : ', loss3)

# y_predict3 = model3.predict(x_test)
# r2 = r2_score(y_test, y_predict3) 
# print('r2스코어 : ', r2)

'''
======================= 1. 기본 출력 =========================
[loss] :  24.64003562927246   
[r2스코어] :  0.7512452089270559

[loss] :  21.36935043334961
[r2스코어] :  0.7842645976879333

[loss] :  19.14341163635254
[r2스코어] :  0.8067366889950223

[loss] :  14.10335636138916
[r2스코어] :  0.8576188392955144

#### 과소적합같아 드롭아웃 비율을 낮추고 층 추가, 에폭과 배치사이즈를 줄이니 loss가 줄었다.
''' 