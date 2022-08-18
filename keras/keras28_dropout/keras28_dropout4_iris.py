from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
import numpy as np
from pandas import get_dummies
import time


#1.데이터 로드 및 정제
datasets = load_iris()
x = datasets.data
y = datasets.target

y = get_dummies(y)  # 원핫인코딩 -> 분류모델

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49) 

#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)   
x_test = scaler.transform(x_test)    

#2. 모델링
model = Sequential()
model.add(Dense(70, input_dim=4))    
model.add(Dense(55))   
model.add(Dropout(0.4))
model.add(Dense(40, activation='relu')) 
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(10, activation='relu')) 
model.add(Dense(3, activation='softmax'))  # 'softmax' -> 분류모델

ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 'categorical_crossentropy' -> 다중분류
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=False)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras28_4_iris{krtime}_MCP.hdf5')
model.fit(x_train, y_train, epochs=10000, batch_size=1, validation_split=0.11111111, callbacks=[es])
#model.save(f"./_save/keras28_4_save_iris{krtime}.h5")


#4. 평가 예측
print("======================= 1. 기본 출력 =========================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

# print("======================= 2. load_model 출력 ======================")
# model2 = load_model(f"./_save/keras28_4_save_iris{krtime}.h5")
# loss2 = model2.evaluate(x_test, y_test)
# print('loss2 : ', loss[0])
# print('accuracy2 : ', loss[1])

# print("====================== 3. mcp 출력 ============================")
# model3 = load_model(f'./_ModelCheckPoint/keras28_4_iris{krtime}_MCP.hdf5')
# loss3 = model3.evaluate(x_test, y_test)
# print('loss3 : ', loss[0])
# print('accuracy3 : ', loss[1])


'''
======================= 1. 기본 출력 =========================
loss :  0.0005985068273730576
accuracy :  1.0
'''