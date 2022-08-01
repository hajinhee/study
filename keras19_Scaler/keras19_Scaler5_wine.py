from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from pandas import get_dummies

#1.데이터 로드 및 정제

datasets = load_wine()
x = datasets.data
y = datasets.target

y = get_dummies(y)
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49) 

#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
scaler.fit(x_train)       
x_train = scaler.transform(x_train)   
x_test = scaler.transform(x_test)    


#2. 모델구성,모델링
model = Sequential()
model.add(Dense(120, activation='linear', input_dim=13))    
model.add(Dense(100 ,activation='relu')) #  
model.add(Dense(80))
model.add(Dense(60 ,activation='relu'))  # 
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(3, activation='softmax'))

model.summary()

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 
es = EarlyStopping  
es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=1,validation_split=0.1111111111, callbacks=[es])


#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print('loss : ',)

'''
결과정리            일반레이어      relu
안하고 한 결과 
loss :              0.0819      0.9559
accuracy :          0.9444      0.5556
MinMax
loss :              0.0476      0.0398
accuracy :          1.0000      1.0000
Standard
loss :              0.3126      0.2785
accuracy :          0.8889      0.9444
Robust
loss :              0.0944      0.0859
accuracy :          0.9444      1.0000
MaxAbs
loss :              0.5498      0.0820
accuracy :          0.9444      0.9444
''' 