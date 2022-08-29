from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import numpy as np, time, warnings
from pandas import get_dummies
from tensorflow.keras.utils import to_categorical
from icecream import ic

#1. load data
datasets = fetch_covtype()
x = datasets.data  # (581012, 54)
y = datasets.target  # (581012,)
ic(x.shape, y.shape)
ic(np.unique(y, return_counts=True))  # [1, 2, 3, 4, 5, 6, 7] -> 분류

# one-hot-encoding
# y = to_categorical(y)  # (581012, 8)
y = get_dummies(y)  # (581012, 7)
ic(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66, stratify=y)

'''
stratify=y
classification을 다룰 때 매우 중요한 옵션값으로 stratify=target으로 지정히먄 각각의 class비율을 train/validation에 유지(한 쪽에 쏠려서 분배되는 것을 방지)한다.  
만약 이 옵션을 지정해 주지 않고 classification 문제를 다룬다면, 성능의 차이가 많이 날 수 있다.
'''

# scaling
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=54))    
model.add(Dense(80))
model.add(Dense(60 ,activation='relu')) 
model.add(Dense(40))
model.add(Dense(20, activation='relu')) 
model.add(Dense(7, activation='softmax')) 

#3. compile, train
learning_rate = 0.01           
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
es = EarlyStopping(monitor='val_accuracy', patience=15, mode='max', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5, mode='max', verbose=1, min_lr=0.0001, factor=0.5)   

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.2, callbacks=[reduce_lr, es])
end = time.time() - start

#4. evaluate
loss, acc = model.evaluate(x_test, y_test)
ic(learning_rate, round(loss, 4), round(acc, 4), f'걸린 시간: {round(end, 4)}')


'''
learning_rate: 0.01
loss: 0.2866
acc: 0.885
걸린 시간: 278.1824
'''