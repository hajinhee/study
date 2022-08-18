#ensemble은 행의 개수가 같아야한다. 열의 개수는 달라도 되지만 개수(행은) 같아야한다!

#1. 데이터
import numpy as np

x1 = np.array([range(100), range(301, 401)])       

x1 = np.transpose(x1)

y1 = np.array(range(1001, 1101)) 
y2 = np.array(range(101, 201)) 
y3 = np.array(range(401,501))

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x1,y1,y2,y3,train_size=0.8,random_state=66)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1 = Input(shape=(2))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(7, activation='relu', name='dense3')(dense2)
output1 = Dense(7, activation='relu', name='output1')(dense3) 

#2-2 output모델1
output21 = Dense(7)(output1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)

#2-3 output모델2
output31 = Dense(7)(output1)
output32 = Dense(21)(output31)
output33 = Dense(21)(output32)
output34 = Dense(11, activation='relu')(output33)
last_output2 = Dense(1)(output34)

#2-4 output모델3
# output41 = Dense(7)(output1)
# output42 = Dense(21)(output41)
# output43 = Dense(21)(output42)
# output44 = Dense(11, activation='relu')(output43)
# last_output3 = Dense(1)(output44)

model = Model(inputs=[input1], outputs=[last_output1,last_output2])#,last_output3

#model.summary()

#3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='mse',optimizer='adam', metrics = 'mae')     # 훈련에 영향을 미치진 않지만, 다른 지표를 볼수있음.
es = EarlyStopping(monitor="val_loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x1_train,[y1_train,y2_train], epochs=10000, batch_size=1, validation_split=0.2,callbacks=[es])  #,y3_train

#4.평가, 예측 
from sklearn.metrics import r2_score
loss = model.evaluate(x1_test,[y1_test,y2_test])    #,y3_test

print('*'*30 + '출력값들' + '*'*30)

y_predict = np.array(model.predict(x1_test)) 
print(y_predict.shape)
#print(y_predict)

#y_predict = (y_predict).reshape(3,20)
#print(y_predict.shape)
#print(y_predict)

r2_1 = r2_score(y1_test,y_predict[0])
r2_2 = r2_score(y2_test,y_predict[1])
#r2_3 = r2_score(y3_test,y_predict[2])

print(r2_1,r2_2)    #,r2_3



