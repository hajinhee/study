import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터

#x = np.array(1,2,3,4,5,6,7)
x = np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]])     
y = np.array([4,5,6,7])                             

print(x.shape, y.shape)        

x = x.reshape(4,1,4)    

#print(x)

#2. 모델구성
model = Sequential()
#model.add(SimpleRNN(5, activation='linear',input_shape=(1,4))) #행은 넣어주지않는다.(row값)                        
model.add(SimpleRNN(10,input_length=3,input_dim=1))
model.add(Dense(4))                 
model.add(Dense(2))                 
model.add(Dense(1))                         

model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #      unit -> n   simplernn 제일처음의 값을 unit이라고한다. 정수의 양수값을 넣는다.
=================================================================
simple_rnn (SimpleRNN)       (None, 10)                120          n * n + n * features + n * bias(1) -> n * (n + features + 1)
_________________________________________________________________
dense (Dense)                (None, 4)                 44           RNN -> Recurrent 뉴럴 네트워크
_________________________________________________________________       
dense_1 (Dense)              (None, 2)                 10           RNN레이어층을 지날때 레이어를 자기들끼리 다시 연산한다.(회귀x) 레이어 한층을 묶음으로 보고 묶음 * 묶음
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 3            tanh...??
=================================================================
Total params: 177
Trainable params: 177
Non-trainable params: 0
_________________________________________________________________
'''

'''
#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam') #mae도있다.
es = EarlyStopping(monitor="loss", patience=500, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x,y, epochs=10000, batch_size=1, callbacks=[es])  

#4. 평가, 예측

model.evaluate(x,y)
y_pred = np.array([5,6,7]).reshape(1,3,1)
result = model.predict(y_pred)    
print(result)
'''
