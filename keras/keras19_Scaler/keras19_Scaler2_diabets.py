from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import numpy as np
from icecream import ic

#1.데이터 로드 및 정제
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42) 

scaler = MaxAbsScaler()  # -1~1 사이로 재조정한다. 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.
x_train = scaler.fit_transform(x_train)   
x_test = scaler.transform(x_test)    


#2. 모델구성,모델링
model = Sequential()
model.add(Dense(100, input_dim=10))
model.add(Dense(80,activation='relu')) 
model.add(Dense(100))
model.add(Dense(150, activation='relu'))
model.add(Dense(120))
model.add(Dense(120))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(50, activation='relu')) 
model.add(Dense(30))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(15))
model.add(Dense(1))
model.summary()


#3. 컴파일 훈련
model.compile(loss='mse', optimizer='adam') 
es = EarlyStopping(monitor="val_loss", patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=10000, batch_size=10, validation_split=0.1, callbacks=[es])


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)


# epochs=500, batch=1  [loss] :  2389.1128  [r2스코어] :  0.5296017874549264
# epochs=10000, batch=10  [loss] :  3232.296875  [r2스코어] :  0.4715778025866174
