from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np                                       

def split_x(dataset, size):                            
    aaa = []                                            
    for i in range(len(dataset) - size + 1):            
        subset = dataset[i : (i + size)]                
        aaa.append(subset)                              
    return np.array(aaa)                                

#1. 데이터로드 및 정제
load_data = np.array(range(1, 101))  # [1, 2, 3, 4, 5, ... , 100] (100, )
x_predict = np.array(range(96,106))  # [ 96  97  98  99 100 101 102 103 104 105] (10, )  

use_data = split_x(load_data, 5)  # RNN에 사용할 수 있도록 연속형 데이터로 변환     
x = use_data[:, :4]  # 행전체와 0~3열               
y = use_data[:, 4]   # 행전체와 4열
real_x_predict = split_x(x_predict, 4)  # RNN에 사용할 수 있도록 연속형 데이터로 변환
print(real_x_predict.shape, real_x_predict)

# DNN은 3차원으로 바꿔줄 필요가 없다. 
#2. 모델링
model = Sequential()   
model.add(Dense(60, input_dim=4, activation='relu'))
model.add(Dense(40))                                                
model.add(Dense(20, activation='relu'))        
model.add(Dense(10))                
model.add(Dense(5))  
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor='loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x, y, epochs=1000, batch_size=1, verbose=1,callbacks=[es])  

#4. 평가, 예측
model.evaluate(x, y)
result = model.predict(real_x_predict)  # (7, 4)
'''
 [[ 96  97  98  99]
 [ 97  98  99 100]
 [ 98  99 100 101]
 [ 99 100 101 102]
 [100 101 102 103]
 [101 102 103 104]
 [102 103 104 105]]
'''
print('[result]: ',  result)
'''
epochs=1000, batch_size=1
[loss]:    9.7418e-07
[result]:  [[ 99.99924]
            [100.99923]
            [101.99921]
            [102.99921]
            [103.99921]
            [104.99918]
            [105.99917]]
'''