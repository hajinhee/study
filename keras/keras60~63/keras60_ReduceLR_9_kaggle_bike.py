from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
import numpy as np,pandas as pd,time
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test,y_predict))

#1. 데이터 로드 및 정제
path = '../Project/Kaggle_Project/bike/'
train = pd.read_csv(path + 'train.csv')    
test_file = pd.read_csv(path + 'test.csv')                   
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x = train.drop(['casual','registered','count'], axis=1)  
x['datetime'] = pd.to_datetime(x['datetime'])
x['year'] = x['datetime'].dt.year
x['month'] = x['datetime'].dt.month
x['day'] = x['datetime'].dt.day
x['hour'] = x['datetime'].dt.hour
x = x.drop('datetime', axis=1)
y = train['count']  
y = np.log1p(y)

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=49)  

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델링
model = Sequential()
model.add(Dense(16, input_dim=12))    
model.add(Dense(24)) #, activation='relu'
model.add(Dense(32)) #, activation='relu'
model.add(Dense(24)) 
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. 컴파일,훈련
learning_rate = 0.001           
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer, metrics='mae')

es = EarlyStopping(monitor="val_loss", patience=15, mode='min',verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto',verbose=1,min_lr=0.0001,factor=0.5)   

start = time.time()
model.fit(x_train,y_train,epochs=1000, batch_size=5,validation_split=0.2, callbacks=[reduce_lr,es])
end = time.time() - start

result = model.evaluate(x_test,y_test,batch_size=32)
print(result)

y_predict = model.predict(x_test)
r2 = r2_score(np.round(np.expm1(y_test)),np.round(np.expm1(y_predict)))
rmse = RMSE(np.round(np.expm1(y_test)),np.round(np.expm1(y_predict)))

print(f"r2는 {r2}, rmse는 {rmse}")

# r2 0.29 -> ReduceLR r2는 0.08395899952905561, rmse는 263.6771642513356
# 아마도 예전에 했던 r2는 exmp안해준 값일거 같다.