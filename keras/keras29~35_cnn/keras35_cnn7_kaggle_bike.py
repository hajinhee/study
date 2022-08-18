from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D,Input
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time

#def RMSE(y_test, y_pred):
#    return np.sqrt(mean_squared_error(y_test,y_predict))    

#1. 데이터 로드 및 정제
path = "../_data/kaggle/bike/"   

train = pd.read_csv(path + 'train.csv')                 
test_file = pd.read_csv(path + 'test.csv')                  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')     

x = train.drop(['datetime','casual','registered','count'], axis=1)  
test_file = test_file.drop(['datetime'], axis=1)

y = train['count']  

#print(np.unique(y, return_counts=True))        # 그냥 졸라 매우 많다. 회귀모델
#y = np.log1p(y) 

#print(x.shape)          # (10886, 8)
#print(y.shape)          # (10886,)
#print(test_file.shape)   # (6493, 8)           


x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)  

scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler()
    
x_train = scaler.fit_transform(x_train).reshape(len(x_train),2,2,2)
x_test = scaler.transform(x_test).reshape(len(x_test),2,2,2)  
test_file = scaler.transform(test_file).reshape(len(test_file),2,2,2)


#2. 모델링

input1 = Input(shape=(2,2,2))
conv1  = Conv2D(4,kernel_size=(2,2),strides=1,padding='valid',activation='relu')(input1) # 2,2,2
conv2  = Conv2D(4,kernel_size=(2,2),strides=1,padding='same',activation='relu')(conv1) # 1,1,1
fla    = Flatten()(conv2)
dense1 = Dense(16,activation="relu")(fla) #
dense2 = Dense(24)(dense1)
drop1  = Dropout(0.2)(dense2)
dense3 = Dense(32,activation="relu")(drop1) # 
output1 = Dense(1)(dense3)
model = Model(inputs=input1,outputs=output1)
      

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

es = EarlyStopping(monitor = "val_loss", patience=100, mode='min',verbose=1,restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras26_7_bike{krtime}_MCP.hdf5')

model.fit(x_train,y_train,epochs=5000,batch_size=50, verbose=1,validation_split=0.11111111,callbacks=[es])#,mcp



#4. 평가
print("======================= 1. 기본 출력 =========================")

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)

#rmse = RMSE(y_test,y_predict)
#print("RMSE : ", rmse)
r2s = str(round(r2,4))
model.save(f"./_save/keras32_7_save_bike{r2s}.h5")

# ############################# 제출용 제작 ####################################
results = model.predict(test_file)

submit_file['count'] = results  
submit_file.to_csv(path + 'test.csv', index=False)  



# results = model.predict(test_file)

# results_int = np.argmax(results, axis=1).reshape(-1,1) + 4 

# submit_file['quality'] = results_int

# acc= str(round(loss[1], 4)).replace(".", "_")
# submit_file.to_csv(path+f"result/accuracy_{acc}.csv", index = False)


'''                            y값 로그O (x하면-값때문에 다 리젝당함)                                                  
결과정리                  일반레이어                      relu                  drop+relu          
                                                                                    
안하고 한 결과 
loss값 :                    
R2 :                           
RMSE :                       

MinMax
loss값 :                    
R2 :                          
RMSE :                        

Standard
loss값 :                     
R2 :                          
RMSE :                        

Robust
loss값 :                     
R2 :                         
RMSE :             

MaxAbs
loss값 :           
R2 :               
RMSE :                
'''
