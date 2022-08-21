from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout ,SimpleRNN, LSTM, GRU, Activation, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.datasets import fetch_covtype
import numpy as np
from pandas import get_dummies
from icecream import ic
import pandas as pd

#1. 데이터 로드 및 정제
datasets = fetch_covtype()
x = datasets.data  # (581012, 54)    
y = datasets.target  # (581012,)
ic(np.unique(y, return_counts=True))  # [1, 2, 3, 4, 5, 6, 7], [211840, 283301, 35754, 2747, 9493, 17367, 20510] -> 다중분류, 원핫인코딩

# 판다스로 변환해 컬럼명 추가 및 삭제
x = pd.DataFrame(x, columns=datasets.feature_names)
x['ydata'] = y
ic(x.corrwith(x['ydata']))
'''
Elevation                            -0.269554
Aspect                                0.017080
Slope                                 0.148285
Horizontal_Distance_To_Hydrology     -0.020317
Vertical_Distance_To_Hydrology        0.081664
Horizontal_Distance_To_Roadways      -0.153450
Hillshade_9am                        -0.035415
Hillshade_Noon                       -0.096426
Hillshade_3pm                        -0.048290
Horizontal_Distance_To_Fire_Points   -0.108936
Wilderness_Area_0                    -0.203913
Wilderness_Area_1                    -0.048059
Wilderness_Area_2                     0.066846
Wilderness_Area_3                     0.323200
Soil_Type_0                           0.090828
Soil_Type_1                           0.118135
Soil_Type_2                           0.068064
Soil_Type_3                           0.099672
Soil_Type_4                           0.077890
Soil_Type_5                           0.112958
Soil_Type_6                          -0.000496
Soil_Type_7                          -0.003667
Soil_Type_8                          -0.006110
Soil_Type_9                           0.243876
Soil_Type_10                          0.035379
Soil_Type_11                         -0.023601
Soil_Type_12                          0.024404
Soil_Type_13                          0.065562
Soil_Type_14                          0.006425
Soil_Type_15                          0.009844
Soil_Type_16                          0.090582
Soil_Type_17                          0.007390
Soil_Type_18                         -0.036452
Soil_Type_19                         -0.028665
Soil_Type_20                         -0.025400
Soil_Type_21                         -0.141746
Soil_Type_22                         -0.135055
Soil_Type_23                         -0.068746
Soil_Type_24                         -0.006449
Soil_Type_25                         -0.000375
Soil_Type_26                         -0.014407
Soil_Type_27                         -0.001702
Soil_Type_28                         -0.124933
Soil_Type_29                         -0.010436
Soil_Type_30                         -0.065347
Soil_Type_31                         -0.075562
Soil_Type_32                         -0.062502
Soil_Type_33                          0.004643
Soil_Type_34                          0.080315
Soil_Type_35                          0.025397
Soil_Type_36                          0.080271
Soil_Type_37                          0.160170
Soil_Type_38                          0.155668
Soil_Type_39                          0.128351
ydata                                 1.000000
'''
x.drop(['Soil_Type_27', 'Soil_Type_33', 'Soil_Type_6', 
        'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_14', 
        'Soil_Type_15', 'Soil_Type_17', 'Soil_Type_24',
        'Soil_Type_25', 'ydata'], axis=1, inplace=True)  
ic(x.shape)  # (581012, 44) 

# 이후 작업을 위해 다시 numpy로 변환 -> 3차원으로 변환
x = x.to_numpy().reshape(len(x), 11, 4)

# 라벨 원핫인코딩
y = get_dummies(y)
ic(y.shape)  # (581012, 7)

# 데이터셋 분리 
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)
ic(x_train.shape)  # (464809, 11, 4)
ic(y_train.shape)  # (464809, 7)
ic(x_test.shape)  # (116203, 11, 4)
ic(y_test.shape)  # (116203, 7)

# 스케일링
# scaler = StandardScaler()  
scaler = MinMaxScaler()   
# scaler = RobustScaler()   
# scaler = MaxAbsScaler() 

# RNN 사용 시 (3차원 -> 2차원 -> 3차원)
x_train = scaler.fit_transform(x_train.reshape(len(x_train), -1)).reshape(x_train.shape)  # (464809, 11, 4)
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
model.add(Dense(7, activation='softmax')) 

#3. 컴파일, 훈련
start = time.time()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
# es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=50, batch_size=100000, validation_split=0.2, verbose=1)    
end = time.time() - start
print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('[loss]:', round(loss[0], 4))
print('[accuracy]: ', round(loss[1], 4))

'''
*********RNN********
epochs=10
[loss]: 1.2211
[accuracy]:  0.4735

epochs=50
[loss]: 0.9092
[accuracy]:  0.6443

*******Comv1D*******
epochs=50
[loss]: 0.7093
[accuracy]:  0.7132
'''