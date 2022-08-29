import numpy as np, pandas as pd, warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
warnings.filterwarnings(action='ignore')

#1. load data 
x = np.array([1,2,3,4,5,6,7,8,9,10])  # (10, )
y = np.array([1,3,5,4,7,6,7,11,9,7])  # (10, )

#2. modeling
model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. compile, train
learning_rate = 0.1  # default=0.001
optimizer_list = [Adam, Adadelta, Adagrad, Adamax, RMSprop, SGD, Nadam]  

for optimizer in optimizer_list:
    optiname = str(optimizer).split('.')[3].split("'")[0]
    model.compile(loss='mse', optimizer=optimizer(learning_rate=learning_rate))
    model.fit(x, y, epochs=10, batch_size=1, verbose=False)

    #4. evaluate, predict
    loss = model.evaluate(x, y, batch_size=1, verbose=False)
    y_predict = model.predict([11])

    print(f'optimizer_name: {optiname} loss: {round(loss,4)} learning_rete: {learning_rate} result: {y_predict}')
    
# 각 optimizer마다 조금씩 방법이 달라서 learning_rate 값을 조금씩 수정해가며 최적의 값을 찾아야한다.