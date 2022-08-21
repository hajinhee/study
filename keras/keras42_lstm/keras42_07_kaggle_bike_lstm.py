from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout ,SimpleRNN, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import r2_score
from icecream import ic
import numpy as np

#1. 데이터 로드 및 정제
path = 'keras/data/kaggle_bike/' 
train = pd.read_csv(path + 'train.csv')                 
test_file = pd.read_csv(path + 'test.csv')                  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')     

ic(train.columns)
'''
['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
'''
ic(train.corrwith(train['count']))
'''
season        0.163439
holiday      -0.005393
workingday    0.011594
weather      -0.128655
temp          0.394454
atemp         0.389784
humidity     -0.317371
windspeed     0.101369
casual        0.690414
registered    0.970948
count         1.000000
'''

x = train.drop(['datetime', 'holiday', 'registered', 'count'], axis=1)  # (10886, 8)
y = train['count']  # (10886, )
ic(np.unique(y, return_counts=True))  # 다양하게 많은 값 -> 회귀모델

test_file.drop(['datetime'], axis=1, inplace=True)
# ic(np.unique(y, return_counts=True))  # 값이 다양함 --> 회귀모델

# RNN에 사용하기 위해 3차원 데이터로 변환
x = x.to_numpy()
x = x.reshape(len(x), 4, 2) 

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)
ic(x_train.shape)  # (8708, 4, 2)

# 스케일링
# scaler = StandardScaler()   
scaler = MinMaxScaler()   
# scaler = RobustScaler()   
# scaler = MaxAbsScaler()   

# RNN 사용 시 
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)  # (8708, 4, 2) -> (8708, 8) -> (8708, 4, 2)
x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(x_test.shape)

# DNN 사용 시
# x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1))
# x_test = scaler.transform(x_test.reshape(len(x_test), -1))

#2. 모델링  
model = Sequential()
model.add(SimpleRNN(10, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))      
model.add(LSTM(10, return_sequences=True, activation='relu'))  
model.add(GRU(10, return_sequences=False, activation='relu'))  
# model.add(Dense(50, input_dim= x.shape[1]))                                                               
model.add(Dense(64))                                                                       
model.add(Dense(32))
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(4))
model.add(Dense(1, activation='linear'))    

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')    
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=500, batch_size=1000, validation_split=0.2, verbose=1, callbacks=[es])     

#4. 평가, 예측      
loss = model.evaluate(x_test, y_test)
print('[loss]: ', round(loss, 4))

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('[r2_score]: ', round(r2, 4))
ic(y_test.shape)  # (2178,)
ic(y_predict.shape)  # (2178, 1)
'''
[loss]:  11563.0723
[r2_score]:  0.6579
'''
