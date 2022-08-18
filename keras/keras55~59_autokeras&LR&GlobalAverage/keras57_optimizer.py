import numpy as np, pandas as pd, warnings

warnings.filterwarnings(action='ignore')

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,3,5,4,7,6,7,11,9,7])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000,input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
# SGD에서 다 파생된 optimizer이다. SGD가 default

learning_rate = 0.1

# optimizer = Adam(learning_rate=0.00001)        # 드디어 쓰기 시작한다. learning_rate. Default 0.001
optimizer_list = [Adam,Adadelta,Adagrad,Adamax,RMSprop,SGD,Nadam]

for optimizer in optimizer_list:

    optiname = str(optimizer).split('.')[3].split("'")[0]

    # model.compile(loss='mse',optimizer='adam', metrics=['mae'])     # 이게 기존 방식
    model.compile(loss='mse', optimizer=optimizer(learning_rate=learning_rate))

    model.fit(x,y, epochs=100, batch_size=1,verbose=False)

    #4. 평가, 예측
    loss = model.evaluate(x,y, batch_size=1,verbose=False)
    y_predict = model.predict([11])

    print(f"현재 옵티마이저 : {optiname} loss : {round(loss,4)} learning_rete : {learning_rate}결과물 : {y_predict}")
    
# 각 optimizer마다 조금씩 방법이 달라서 learning_rate 값을 조금씩 수정해가며 최적의 값을 찾아야한다.