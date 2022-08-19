import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D,Input
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time
from icecream import ic

def RMSE(y_test, y_pred):
   return np.sqrt(mean_squared_error(y_test, y_pred))    

#1. 데이터 로드 및 정제
path = 'keras/data/kaggle_bike/' 
train = pd.read_csv(path + 'train.csv')                 
test_file = pd.read_csv(path + 'test.csv')                  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')     

x = train.drop(['datetime', 'casual', 'registered', 'count'], axis=1)  # (10886, 8)
y = train['count']  # (10886,)
test_file.drop(['datetime'], axis=1, inplace=True)  # (6493, 8)
ic(np.unique(y, return_counts=True))  # 회귀모델
# ic(x.shape)         
# ic(y.shape)          
# ic(test_file.shape)          

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49)  

# scaler = MinMaxScaler()   
# scaler = StandardScaler() 
# scaler = RobustScaler()   
scaler = MaxAbsScaler()   
x_train = scaler.fit_transform(x_train).reshape(len(x_train), 2, 2, 2)  # (9797, 2, 2, 2)
x_test = scaler.transform(x_test).reshape(len(x_test), 2, 2, 2)  # (1089, 2, 2, 2)
test_file = scaler.transform(test_file).reshape(len(test_file), 2, 2, 2)  # (6493, 2, 2, 2)

#2. 모델링
input1 = Input(shape=(2, 2, 2))
conv1 = Conv2D(50, kernel_size=(2, 2), strides=1, padding='valid', activation='relu')(input1)
conv2 = Conv2D(30, kernel_size=(2, 2), strides=1, padding='same', activation='relu')(conv1)
conv3 = Conv2D(20, kernel_size=(2, 2), strides=1, padding='same', activation='relu')(conv2)
conv4 = Conv2D(10, kernel_size=(2, 2), strides=1, padding='same', activation='relu')(conv3)
fla = Flatten()(conv4)
dense1 = Dense(50, activation='relu')(fla) 
dense2 = Dense(40)(dense1)
drop1 = Dropout(0.5)(dense2)
dense3 = Dense(20, activation='relu')(drop1) 
drop2 = Dropout(0.5)(dense3)
output1 = Dense(1)(drop2)
model = Model(inputs=input1, outputs=output1)
      
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor = 'val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_7_bike{krtime}_MCP.hdf5')
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=32, validation_split=0.2, callbacks=[es])

#4. 평가
print('======================= 1. 기본 출력 =========================')
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

rmse = RMSE(y_test, y_predict)
print('RMSE : ', rmse)
r2s = str(round(r2,4))
model.save(f'./_save/keras32_7_save_bike{r2s}.h5')

############################## 제출용 제작 ####################################
results = model.predict(test_file)
submit_file['count'] = results  
path = 'keras/save/kaggle_bike/' 
submit_file.to_csv(path + 'test.csv', index=False)  


'''              
'''
