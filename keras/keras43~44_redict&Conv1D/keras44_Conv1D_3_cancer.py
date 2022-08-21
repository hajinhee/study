from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Activation, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.metrics import r2_score
from pandas import get_dummies
from icecream import ic
import pandas as pd

#1. 데이터 로드 및 정제
datasets = load_breast_cancer()
x = datasets.data  # (569, 30)
y = datasets.target # (569, )
ic(np.unique(y, return_counts=True))  # array([0, 1]), array([212, 357]   이진분류 -> 원핫인코딩

# 판다스로 변환해 컬럼명 추가 및 삭제
x = pd.DataFrame(x, columns=datasets.feature_names)
x['ydata'] = y
ic(x.corrwith(x['ydata']))
'''
mean radius               -0.730029
mean texture              -0.415185
mean perimeter            -0.742636
mean area                 -0.708984
mean smoothness           -0.358560
mean compactness          -0.596534
mean concavity            -0.696360
mean concave points       -0.776614
mean symmetry             -0.330499
mean fractal dimension     0.012838
radius error              -0.567134
texture error              0.008303
perimeter error           -0.556141
area error                -0.548236
smoothness error           0.067016
compactness error         -0.292999
concavity error           -0.253730
concave points error      -0.408042
symmetry error             0.006522
fractal dimension error   -0.077972
worst radius              -0.776454
worst texture             -0.456903
worst perimeter           -0.782914
worst area                -0.733825
worst smoothness          -0.421465
worst compactness         -0.590998
worst concavity           -0.659610
worst concave points      -0.793566
worst symmetry            -0.416294
worst fractal dimension   -0.323872
ydata                      1.000000
'''
x.drop(['symmetry error', 'texture error', 'ydata'], axis=1, inplace=True)  
ic(x.shape)  # (569, 28)

# 이후 작업을 위해 다시 numpy로 변환 -> 3차원으로 변환
x = x.to_numpy().reshape(len(x), 7, 4)  # (569, 7, 4)

# 라벨 원핫인코딩
y = get_dummies(y)
ic(y.shape)  # (569, 2)

# 데이터셋 분리 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)
ic(x_train.shape)  # (455, 7, 4)
ic(y_train.shape)  # (455, 2)
ic(x_test.shape)  # (114, 7, 4)
ic(y_test.shape)  # (114, 2)

# 스케일링
# scaler = StandardScaler()  
scaler = MinMaxScaler()   
# scaler = RobustScaler()   
# scaler = MaxAbsScaler() 

# RNN 사용 시 (3차원 -> 2차원 -> 3차원)
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)  # (455, 7, 4)
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
model.add(Dense(2, activation='sigmoid')) 

#3. 컴파일, 훈련
start = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  
# es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1)    
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('[loss]:', round(loss[0], 4))
print('[accuracy]: ', round(loss[1], 4))

'''
*******RNN******
[loss]: 0.1971
[accuracy]:  0.9386


*****Comv1D*****
[loss]: 0.2691
[accuracy]:  0.9474
'''