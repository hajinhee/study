from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Activation, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.datasets import load_wine
import numpy as np
from pandas import get_dummies
from icecream import ic
import pandas as pd

#1.데이터로드 및 정제
datasets = load_wine()
x = datasets.data  # (178, 13)
y = datasets.target  # (178,)
ic(np.unique(y,return_counts=True))  # array([0, 1, 2]), array([59, 71, 48] -> 다중분류 -> 원핫인코딩
ic(y.shape)

# 판다스로 변환해 컬럼명 추가 및 삭제
x = pd.DataFrame(x, columns=datasets.feature_names)
x['ydata'] = y
ic(x.corrwith(x['ydata']))
'''
alcohol                        -0.328222
malic_acid                      0.437776
ash                            -0.049643
alcalinity_of_ash               0.517859
magnesium                      -0.209179
total_phenols                  -0.719163
flavanoids                     -0.847498
nonflavanoid_phenols            0.489109
proanthocyanins                -0.499130
color_intensity                 0.265668
hue                            -0.617369
od280/od315_of_diluted_wines   -0.788230
proline                        -0.633717
ydata                           1.000000
'''
x.drop(['ash', 'ydata'], axis=1, inplace=True) 
ic(x.shape)  # (178, 12)

# 이후 작업을 위해 다시 numpy로 변환 -> 3차원으로 변환
x = x.to_numpy().reshape(len(x), 4, 3)  # (178, 4, 3)

# 라벨 원핫인코딩
y = get_dummies(y)  # (178, 3)

# 데이터셋 분리 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)
ic(x_train.shape)  # (142, 4, 3)
ic(y_train.shape )  # (142, 3)
ic(x_test.shape)  # (36, 4, 3)
ic(y_test.shape)  # (36, 3)

# 스케일링
# scaler = StandardScaler()  
scaler = MinMaxScaler()   
# scaler = RobustScaler()   
# scaler = MaxAbsScaler() 

# RNN 사용 시 (3차원 -> 2차원 -> 3차원)
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)  # (142, 4, 3)
x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(x_test.shape)

# DNN 사용 시 (3차원 -> 2차원)
# x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1))
# x_test = scaler.transform(x_test.reshape(len(x_test),-1))

#2. 모델링  
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))  
# model.add(LSTM(10, return_sequences=True, activation='relu'))    
# model.add(GRU(10, return_sequences=False, activation='relu'))  
# model.add(Dense(50, input_dim=x.shape[1]))                    
model.add(Conv1D(10, 2, input_shape=(x_train.shape[1], x_train.shape[2])))  
model.add(Flatten()) 
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu'))
model.add(Dense(4))
model.add(Dense(3, activation='softmax')) 

#3. 컴파일, 훈련
start = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
# es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1)    
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('[loss]:', round(loss[0], 4))
print('[accuracy]: ', round(loss[1], 4))

'''
*******RNN*******
[loss]: 0.0703
[accuracy]:  0.9722

*****Comv1D*****
[loss]: 0.0314
[accuracy]:  0.9722
'''