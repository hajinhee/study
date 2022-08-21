from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Activation,Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
import numpy as np
from icecream import ic
import pandas as pd

#1. 데이터 로드 및 정제
datasets = load_diabetes()
x = datasets.data  # (442, 10)
y = datasets.target 
ic(np.unique(y,return_counts=True))  # 다양한 값 -> 회귀모델

# 판다스로 변환해 컬럼명 추가 및 삭제
x = pd.DataFrame(x, columns=datasets.feature_names)
x['ydata'] = y
ic(x.corrwith(x['ydata']))
'''
age      0.187889
sex      0.043062
bmi      0.586450
bp       0.441482
s1       0.212022
s2       0.174054
s3      -0.394789
s4       0.430453
s5       0.565883
s6       0.382483
ydata    1.000000
'''
x.drop(['sex', 'ydata'], axis=1, inplace=True)  # (442, 9)
ic(x.shape)            

# 이후 작업을 위해 다시 numpy로 변환
x = x.to_numpy()

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)
ic(x_train.shape)  # (353, 9)
ic(x_test.shape)  # (89, 9)

# 스케일링
# scaler = StandardScaler()  
scaler = MinMaxScaler()   
# scaler = RobustScaler()   
# scaler = MaxAbsScaler()   

# RNN 사용 시 (2차원 -> 3차원)
# x_train = scaler.fit_transform(x_train).reshape(len(x_train), 3, 3)  # (353, 9) -> (353, 3, 3)
# x_test = scaler.transform(x_test).reshape(len(x_test), 3, 3)  # (89, 9) -> (89, 3, 3)

# Conv1D 사용 시 (2차원 -> 3차원)
x_train = scaler.fit_transform(x_train).reshape(len(x_train), 3, 3)
x_test = scaler.transform(x_test).reshape(len(x_test), 3, 3)

#2.모델링   각 데이터에 알맞게 튜닝
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))     
# model.add(LSTM(10, return_sequences=True, activation='relu')) 
# model.add(GRU(10, return_sequences=False, activation='relu'))   
model.add(Conv1D(10, 2, input_shape=(x_train.shape[1], x_train.shape[2])))  
model.add(Flatten())      
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(4))
model.add(Dense(1))  

#3. 컴파일, 훈련
start = time.time()
model.compile(loss='mse', optimizer='adam') 
# es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1)       
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측  
loss = model.evaluate(x_test, y_test)
print('[loss]: ', round(loss, 4))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('[r2_score]: ', round(r2, 4))

'''
**********RNN*********
[loss]:  2197.6758
[r2_score]:  0.5875

********Conv1D********
[loss]:  2159.176
[r2_score]:  0.5948
'''