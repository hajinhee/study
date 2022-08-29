from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import numpy as np,time,warnings
from icecream import ic

#1. load data
datasets = load_boston()
x = datasets.data  #  (506, 13)
y = datasets.target  # (506,)
ic(x.shape, y.shape)
ic(np.unique(y, return_counts=True))  # 회귀

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling
model = Sequential()
model.add(Dense(20, input_dim=13))
model.add(Dense(25))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(35))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. compile, train
learning_rate = 0.01           
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer, metrics='mae') 
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, min_lr=0.0001, factor=0.5)   

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2, callbacks=[reduce_lr, es])
end = time.time() - start

#4. evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=32)
ic(loss, mae)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
ic(r2)


'''
loss: 22.581817626953125, mae: 3.275252342224121
r2: 0.7298274143628392
'''