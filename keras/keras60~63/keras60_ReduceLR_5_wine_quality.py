# csv파일 white
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import numpy as np,time,warnings, pandas as pd
from pandas import get_dummies

#1. 데이터 로드

path = 'D:\_data/'

datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', index_col=None, header=0)
datasets = datasets.to_numpy()

x = datasets[:, :11]    
y = datasets[:, 11] 

y = get_dummies(y)

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성,모델링
model = Sequential()
model.add(Dense(120, activation='linear', input_dim=11))    
model.add(Dense(100 ,activation='relu')) #  
model.add(Dense(80))
model.add(Dense(60 ,activation='relu'))  # 
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(7, activation='softmax'))

#3. 컴파일 훈련
learning_rate = 0.001           
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 

es = EarlyStopping(monitor="val_accuracy", patience=15, mode='max',verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=5, mode='max',verbose=1,min_lr=0.0001,factor=0.5)   

start = time.time()
model.fit(x_train,y_train,epochs=1000, batch_size=1000,validation_split=0.2, callbacks=[reduce_lr,es])
end = time.time() - start

#4. 평가
loss , acc = model.evaluate(x_test,y_test)

print(f"lr : {learning_rate}, loss : {round(loss,4)}, acc : {round(acc,4)}, 걸린시간 : {round(end,4)}초")

# lr : 0.001, loss : 1.1418, acc : 0.5367, 걸린시간 : 1.9991초