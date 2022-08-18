import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
import time

#1. 데이터

x = np.array([   [1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12],[20,30,40],[30,40,50],[40,50,60]   ])     

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])                             
        
x = x.reshape(13,3,1)

#2. 모델구성
model = Sequential()            
model.add(GRU(10,activation='relu' ,input_shape=(3,1),return_sequences=False))   # 1도 가능하다.    (N,3,1) -> (N,10)              (none,3,1)
# 연속적으로 LSTM을 쓰려면 또 3차원의 데이터를 넣어줘야한다.                                                                          (none,3,10)
# 그걸 도와주는 옵션이 return_sequence 옵션이다 True설정하면 다음레이어에 3차원값을 그대로준다.                                         (none,3,10)
# 그런데 연속적으로 해봤을때 좋은게 없다. 시계열 데이터를 RNN연산해서 나오는 output값은 연산을 많이 거쳐서 나온 값이므로
# 나온값들은 시계열데이터라고 보기엔 좀 힘들다. 그 특성이 많이 희석되어져서 나오기때문에 많은 기대를 하기 힘들다.
model.add(Dense(60,activation='relu',))
model.add(Dense(40))                                                # Dense는 주는 값을 그대로 받긴하지만 어차피 출력은 2차원으로 해야해서 언젠간 flatten으로 데이터를 펴줘야한다.
model.add(Dense(20,activation='relu',))        
model.add(Dense(10))        
model.add(Dropout(0.1))         
model.add(Dense(5))  
model.add(Dense(1))
#model.summary()


#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam') #mae도있다.
es = EarlyStopping(monitor="loss", patience=500, mode='min',verbose=1,baseline=None, restore_best_weights=True)

start = time.time()
model.fit(x,y, epochs=10000, batch_size=1, verbose=1,callbacks=[es])  
end = time.time() - start

print("걸린시간 : ", round(end, 3), '초')

#4. 평가, 예측

model.evaluate(x,y)

y_pred = np.array([50,60,70]).reshape(1,3,1)

result = model.predict(y_pred)  

print(result)