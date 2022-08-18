# RNN - > DNN으로 변경
'''
id    x1 x2 x3 x4   y
0    [ 1  2  3  4] [5]
1    [ 2  3  4  5] [6]
2    [ 3  4  5  6] [7]
3    [ 4  5  6  7] [8]      로 이해하고

   y = w1x1 + w2x2 + w3x3 + w4x4의 칼럼이 4개인 데이터셋으로 이해하면 
   충분히 DNN방식으로 변경 할 수 있다.

''' 
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
real_x_predict = split_x(x_predict,4)   # usedata로 담아서 x,y나누듯이 use_x_p~~에 연속된 값으로 담아준다.

# DNN은 이 아래부터 바꿔줄 이유가 없다. 이 아래는 삭제한다. 보고 싶으면 split2 확인.

#2. 모델링

model = Sequential()   
model.add(Dense(60,input_dim=4,activation='relu',))
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
