import numpy as np
from tensorflow.keras.models import Sequential, Model # 함수형모델 Model
from tensorflow.keras.layers import Dense, Input


#1. 데이터
x = np.array([range(100), range(301,401), range(1,101)])  # (3,100) 
y = np.array([range(71,81)])  # (1, 10)
print(y.shape)
x = np.transpose(x)  # (100,3)
y = np.transpose(y)  # (10, )
    

#2. 모델구성     
model = Sequential()
model.add(Dense(10, input_dim=3))      
model.add(Dense(8))
model.add(Dense(2))
model.summary()
'''
dense (Dense)                (None, 10)                40
x자리가 None인 이유 = 행의 개수는 신경쓰지 않겠다. 상관없다.
'''

