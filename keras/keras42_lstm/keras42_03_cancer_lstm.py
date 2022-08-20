from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.metrics import r2_score
from pandas import get_dummies
from icecream import ic
import pandas as pd
import pandas as pd

#1. 데이터로드 및 정제
datasets = load_breast_cancer()
x = datasets.data  # (569, 30)
y = datasets.target  # array([0, 1]), array([212, 357] -> 이진분류
ic(np.unique(y, return_counts=True))       

'''
[unique value 추출 방법]
# 방법 1. 
numpy의 np.unique(return_counts=True) ---> ndarray만 가능
ic(np.unique(y, return_counts=True))       

# 방법 2. 
pandas의 value_counts() ---> Dataframe column 혹은 Series만 가능
y = pd.DataFrame(y)  # numpy --> pandas 변환
ic(y.value_counts())  
'''

# 컬럼명 추가하기 위해 pandas로 변환
x = pd.DataFrame(x, columns=datasets.feature_names)
x['ydata'] = y
# ic(x.corrwith(x['ydata']))
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
y = get_dummies(y)  # 원핫인코딩
ic(type(y), y.shape) # (569, 2)

# 불필요한 컬럼 제거
x.drop(['symmetry error','ydata'], axis=1, inplace=True)  
ic(x.shape)  # (569, 29) 

# 이후 작업(원핫인코딩&RNN)을 위해 다시 numpy로 변환 후 reshape 
x = x.to_numpy().reshape(len(x), 1, 29)  # (569, 1, 29)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)

scaler = StandardScaler()   
# scaler = MinMaxScaler()   
# scaler = RobustScaler()   
# scaler = MaxAbsScaler()   

# RNN 사용 시 (3차원 -> 2차원 -> 3차원)
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)  # (455, 1, 29) -> (455, 29) -> (455, 1, 29)
x_test = scaler.transform(x_test.reshape(len(x_test), -1)).reshape(x_test.shape)
# DNN 사용 시 (3차원 -> 2차원)
# x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1))
# x_test = scaler.transform(x_test.reshape(len(x_test), -1))

#2. 모델링 
model = Sequential()
model.add(SimpleRNN(10, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))  # SimpleRNN     
model.add(LSTM(10, return_sequences=True, activation='relu'))  # LSTM                         
model.add(GRU(10, return_sequences=False, activation='relu'))  # GRU  
#model.add(Dense(50, input_dim=x.shape[1]))                                                                
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(4))
model.add(Dense(2, activation='sigmoid'))  # 'sigmoid' -> 이진분류

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.2, verbose=1, callbacks=[es])       

#4. 평가, 예측 --> [평가지표] 분류모델은 'acuuracy', 회귀모델은 'r2' 사용
loss = model.evaluate(x_test, y_test)
ic(round(loss[0], 4))
ic(round(loss[1], 4))


'''
[loss]: 0.1027
[accuracy]: 0.9737
'''
