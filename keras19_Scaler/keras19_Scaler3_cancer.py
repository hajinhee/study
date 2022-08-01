from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.utils import to_categorical  

#1.데이터 로드 및 정제

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

#print(x.shape, y.shape) #(569, 30) (569,)
#print(np.unique(y))    # [0 1]     2개의 값 -> one hot encoding
y = to_categorical(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49) 

scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler() 
x_train = scaler.fit_transform(x_train)    
x_test = scaler.transform(x_test) 


#2. 모델구성,모델링
model = Sequential()
model.add(Dense(30, input_dim=30))    
model.add(Dense(25 ,activation='relu')) #   
model.add(Dense(15 ,activation='relu')) #
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2, activation='sigmoid'))
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 30)                930
_________________________________________________________________
dense_1 (Dense)              (None, 25)                775
_________________________________________________________________
dense_2 (Dense)              (None, 15)                390
_________________________________________________________________
dense_3 (Dense)              (None, 10)                160
_________________________________________________________________
dense_4 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 6
=================================================================
Total params: 2,316
Trainable params: 2,316
Non-trainable params: 0
_________________________________________________________________
'''

#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy']) 
es = EarlyStopping  
es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=5,validation_split=0.1111111, callbacks=[es])


#4. 평가 예측
loss = model.evaluate(x_test,y_test)
print(loss)

'''
결과정리                일반레이어                      relu
안하고 한 결과 
loss :                                          0.17505894601345062
accuracy :                                      0.9298245906829834
MinMax
loss :                                          0.21379293501377106
accuracy :                                      0.9298245906829834
Standard
loss :             
accuracy :         
Robust
loss :             
accuracy :         
MaxAbs
loss :             
accuracy :         
''' 