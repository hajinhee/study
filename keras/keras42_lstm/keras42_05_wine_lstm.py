from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import load_wine
import numpy as np
from pandas import get_dummies
from icecream import ic
import pandas as pd

#1. 데이터 로드 및 정제
datasets = load_wine()
x = datasets.data  # (178, 13)
y = datasets.target  # (178, 3)
ic(np.unique(y, return_counts=True))  # array([0, 1, 2]), array([59, 71, 48] --> 다중분류 -> 원핫인코딩     

# 상관관계 분석 후 컬럼 제거
# 데이터가 ndarray(numpy)인 경우 dataframe(pandas)으로 변환 후 작업한다. 이 때 원핫인코딩은 뒤로 미뤄둔다. 
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

# 불필요한 컬럼 삭제 후 다시 ndarray(numpy)로 변환
x.drop(['ash', 'ydata'], axis=1, inplace=True)  # (178, 12) <-- 변경된 컬럼 수 확인      
x = x.to_numpy()

# 원핫인코딩
y = get_dummies(y)  

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)

# 스케일링
scaler = StandardScaler()   
# scaler = MinMaxScaler()   
# scaler = RobustScaler()   
# scaler = MaxAbsScaler()  

# RNN사용시 
x_train = scaler.fit_transform(x_train).reshape(len(x_train), 4, 3)  # (142, 12) ->  (142, 4, 3)
x_test = scaler.transform(x_test).reshape(len(x_test), 4, 3)
# DNN사용시
# x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1))
# x_test = scaler.transform(x_test.reshape(len(x_test), -1))

#2. 모델링  
model = Sequential()
model.add(SimpleRNN(10, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))       
model.add(LSTM(10, return_sequences=True, activation='relu'))  
model.add(GRU(10, return_sequences=False, activation='relu')) 
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(4))
model.add(Dense(3, activation='softmax'))   

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1, callbacks=[es])        

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('[loss]: ', round(loss[0], 4))
print('[accuracy]: ', round(loss[1], 4))

'''
[loss]:  0.068
[accuracy]:  0.9722
'''