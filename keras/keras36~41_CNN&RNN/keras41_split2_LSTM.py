# 모델 한번 구성해봐~
 
#0. 내가쓸 기능들 import 및 함수 선언  
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

### 데이터 로드
load_data = np.array(range(1,101))
x_predict = np.array(range(96,106))     # 주의 x_pre~~도 형태를 x와 같은 형태로 맞춰줘야한다.
# size 5 -> timestep4, y label 1개

### 데이터 정제
usedata = split_x(load_data,5)          # RNN에 쓸수있게 연속된 데이터형태로바꿔줌.

x = usedata[:,:4]               # 바꾼 데이터를 다시 쪼개서 x와 y로 만들어줌
y = usedata[:,4]
#print(x,y)     # 잘된것 확인.

print(x)
#print(len(x))  # 일일이 몇행인지 셀수 없으니까. x의 행의 개수 체크 96개 확인.
x = x.reshape(len(x),4,1)   # x를 RNN모델로 돌리기 위해 96,4,1의 3차원 형태로 만들어준다.
#print(x.shape) #확인.       
'''
96,4,1은 이런형태
[[[ 1]
  [ 2]
  [ 3]
  [ 4]]

 [[ 2]
  [ 3]
  [ 4]
  [ 5]]

 [[ 3]
  [ 4]
  [ 5]
  [ 6]]     

96,2,2
[[[ 1  2]
  [ 3  4]]

 [[ 2  3]
  [ 4  5]]

 [[ 3  4]
  [ 5  6]]
  
96,1,4
[[[ 1  2  3  4]]

 [[ 2  3  4  5]]
'''
'''
use_x_predict = split_x(x_predict,4)   # usedata로 담아서 x,y나누듯이 use_x_p~~에 연속된 값으로 담아준다.
#print(len(use_x_predict))      # len이용해서 개수 확인 7.

real_x_predict = use_x_predict.reshape(len(use_x_predict),4,1)      # x데이터와 같이 none,4,1로 변환 뒤의 4,1은 유동적.
#print(real_x_predict.shape)


#2. 모델링

model = Sequential()   
model.add(SimpleRNN(10,activation='relu' ,input_shape=(4,1),return_sequences=False))
#model.add(LSTM(10,activation='relu' ,input_shape=(4,1),return_sequences=True))
#model.add(GRU(10,activation='relu' ,input_shape=(4,1),return_sequences=False))
model.add(Dense(60,activation='relu',))
model.add(Dense(40))                                                
model.add(Dense(20,activation='relu',))        
model.add(Dense(10))                
model.add(Dense(5))  
model.add(Dense(1))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor="loss", patience=100, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x,y, epochs=10000, batch_size=1, verbose=1,callbacks=[es])  

#4. 평가, 예측

model.evaluate(x,y)

#real_x_predict로 위에서 바로 값 세팅 해놓음 바로 다음 step~

result = model.predict(real_x_predict)

print(result)
'''