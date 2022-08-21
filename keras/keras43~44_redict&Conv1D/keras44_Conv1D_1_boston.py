from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Activation, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import numpy as np
from icecream import ic
import pandas as pd

#1. 데이터 로드 및 정제 
datasets = load_boston()
x = datasets.data  # (506, 13)
y = datasets.target  # (506,)
ic(y.shape, np.unique(y, return_counts=True))  # 다양한 값 --> 회귀모델     

# 판다스로 변환해 컬럼명 추가 및 삭제
x = pd.DataFrame(x, columns=datasets.feature_names)
x['ydata'] = y
ic(x.corrwith(x['ydata']))
'''
CRIM      -0.388305
ZN         0.360445
INDUS     -0.483725
CHAS       0.175260
NOX       -0.427321
RM         0.695360
AGE       -0.376955
DIS        0.249929
RAD       -0.381626
TAX       -0.468536
PTRATIO   -0.507787
B          0.333461
LSTAT     -0.737663
ydata      1.000000
'''
x.drop(columns='ydata', axis=1, inplace=True)  # (506, 13)

# 이후 작업을 위해 넘파이 변환 및 3차원 변환
x = x.to_numpy().reshape(len(x), 13, 1)   

# 데이터 분리 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)

# 스케일링
# scaler = StandardScaler()  
scaler = MinMaxScaler()  
# scaler = RobustScaler()  
# scaler = MaxAbsScaler()  

# RNN 사용 시 (3d -> 2d -> 3d) 
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)
x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(x_test.shape)

# Conv 사용 시 (3d -> 2d -> 3d)
# x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)
# x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(x_test.shape)

#2. 모델링 
model = Sequential()
model.add(SimpleRNN(10, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))      
model.add(LSTM(10, return_sequences=True, activation='relu'))                            
model.add(GRU(10, return_sequences=False, activation='relu'))    
# model.add(Conv1D(10, 2, input_shape=(x_train.shape[1], x_train.shape[2])))  # Conv1D -> 3d
model.add(Flatten())              
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(15, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(5))
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
*******Conv1D********
[loss]:  21.7814
[r2_score]:  0.7763

*********RNN*********
[loss]:  20.463
[r2_score]:  0.7898

'''