# csv파일 white
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import numpy as np,time,warnings, pandas as pd
from pandas import get_dummies
from icecream import ic

#1. load data
datasets = pd.read_csv('keras/data/kaggle/winequality/winequality-white.csv', sep=';', index_col=None, header=0)  # header: 열 이름(헤더)으로 사용할 행 지정

# pandas.dataframe -> numpy.ndarray 변환
datasets = datasets.to_numpy()

x = datasets[:, :11]  # 'quality' column을 제외한 나머지 데이터 
y = datasets[:, 11]   # 'quality' column 데이터

# check target
ic(np.unique(y, return_counts=True))  # [3., 4., 5., 6., 7., 8., 9.] -> 분류

# one hot encoding
y = get_dummies(y)

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66, stratify=y)  # stratify=y --> classification

# scaling
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling
model = Sequential()
model.add(Dense(120, activation='linear', input_dim=11))    
model.add(Dense(100, activation='relu'))   
model.add(Dense(80))
model.add(Dense(60, activation='relu'))  
model.add(Dense(40))
model.add(Dense(20))
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
loss , acc = model.evaluate(x_test, y_test)
ic(learning_rate, round(loss, 4), round(acc, 4), f'걸린 시간: {round(end, 4)}')


'''
learning_rate: 0.01
loss: 1.0431
acc: 0.5531
걸린 시간: 5.0717
'''