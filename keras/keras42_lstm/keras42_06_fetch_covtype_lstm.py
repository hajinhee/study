#0.내가쓸 기능들 import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout ,SimpleRNN, LSTM, GRU, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping

# 개별 import
from sklearn.datasets import fetch_covtype
import numpy as np
from pandas import get_dummies
#1.데이터로드 및 정제

### 1-1.로드영역    데이터 형태를 x,y로 정의해주세요.
datasets = fetch_covtype()
x = datasets.data           
y = datasets.target    

### 1-2. RNN하기위해 shape변환

#x값 관측.    x의 shape를 기록해주세요.     :   (581012, 54)
#print(x.shape)      

#y값 관측.    y의 shape를 기록해주세요.     :   array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510]  -> 다중분류, 원핫인코딩
y = get_dummies(y)
#numpy      
#print(np.unique(y,return_counts=True))       

#pandas   
#print(y,value_counts())


### 1-3. 상관관계 분석 후 x칼럼제거.        스킵 가능.------------------------------------------------
#데이터가 np일 경우 pandas import해서 변환후 작업. 원핫인코딩 끄고 작업 후 다시 원핫인코딩해주세요.
# import pandas as pd
# x = pd.DataFrame(x, columns=datasets.feature_names)
# x['ydata'] = y
# #print(x.corr())
# x = x.drop(['','ydata'],axis=1)  # drop시킬 column명 기재.
# #print(x.shape)            # 변경된 칼럼개수 확인.  기재 : 
# #그 이후의 작업 계속해주기 위해 numpy로 변환
# x = x.to_numpy()
#---------------------------------------------------------------------------------------------------


### 1-4. x의 shape변환
#x = x.reshape(len(x),9,6)      #len(x)뒤의 영역은 사용자 지정입니다!   DNN모델일 경우 주석처리.


### 1-5. train & test분리 
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=49)


### 1-6. scaler적용. 스킵 가능----------------------------------------------------------------------

scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler()

# RNN사용시 
# 자동으로 3차원데이터를 2차원으로 만들어서 스케일링 적용하고 다시 3차원으로 적용해줌.
#x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1)).reshape(x_train.shape)
#x_test = scaler.transform(x_test.reshape(len(x_test),-1)).reshape(x_test.shape)

# DNN사용시
x_train = scaler.fit_transform(x_train.reshape(len(x_train),-1))
x_test = scaler.transform(x_test.reshape(len(x_test),-1))
#--------------------------------------------------------------------------------------------------


#2.모델링   각 데이터에 알맞게 튜닝
model = Sequential()
#model.add(SimpleRNN(10,input_shape=(x.shape[1],x.shape[2])   ,return_sequences=True))       # 공백안에 ,activation='relu'도 사용해보세요.
#model.add(LSTM(10,return_sequences=True,activation='relu'))   #                             # 두번째, 세번째 줄은 주석처리해서 1개만 사용해보세요.
#model.add(GRU(10,return_sequences=False,activation='relu'))  #  
model.add(Dense(50,input_dim= x.shape[1]))                                                  # DNN방식적용시 위의 RNN주석 걸고 위의 1-4에서 두번째 옵션 선택합니다.                
model.add(Dense(64))                                                                         # DNN방식 사용시 model.add(Dropout(0.5)) 복사후 사용.
model.add(Dense(32))
model.add(Dense(16,activation="relu")) #
model.add(Dense(8,activation="relu")) #
model.add(Dense(4))
model.add(Dense(7,activation = 'softmax'))    # default = 'linear' 이진분류 = 'sigmoid' , 다중분류 = 'softmax' 



#3.컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])    # 회귀모델 = mse, 이진분류 = binary_crossentropy, 다중분류 = categorical_crossentropy, 분류는 ,metrics=['accuracy']
es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train, epochs=10000, batch_size=100000,validation_split=0.2,verbose=1,callbacks=[es])        # batch_size 센스껏 조절!  



#4.평가,예측        회귀모델은 r2,  분류모델은 accuracy

loss = model.evaluate(x_test,y_test)

##분류모델일때 주석 해제.
print("----------------------loss & accuracy-------------------------")
print(round(loss[0],4))
print(round(loss[1],4))

### 회귀모델일때 주석 해제.
# print("----------------------loss값-------------------------")
# print(round(loss,4))
# y_predict = model.predict(x_test)

# print("=====================r2score=========================")
# r2 = r2_score(y_test,y_predict)
# print(round(r2,4))

#5.결과 정리 창

#                   DNN                 |             CNN                |               RNN
#loss:                                                                     
#                                                                    
#                                                                                   RNN + MIN                                                                           
#loss:                                                                                       
#acc:                                                                                  
#              
