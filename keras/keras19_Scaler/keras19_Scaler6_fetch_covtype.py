from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from icecream import ic
import numpy as np
from pandas import get_dummies
from sklearn.metrics import accuracy_score



#1.데이터 로드 및 정제
datasets = fetch_covtype()
x = datasets.data  # (581012, 54)
y = datasets.target  # (581012, 7)


print(y)  # 분류문제
'''
[5 5 2 ... 3 3 3]
'''
print(np.unique(y))
'''
[1 2 3 4 5 6 7]
'''

y = get_dummies(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42) 


scaler = MinMaxScaler()  
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)   
x_test = scaler.transform(x_test)    


#2. 모델구성,모델링
model = Sequential()
model.add(Dense(100, input_dim=54))    
model.add(Dense(80, activation='relu'))
model.add(Dense(70))
model.add(Dense(60, activation='relu')) 
model.add(Dense(40, activation='relu'))
model.add(Dense(30))
model.add(Dense(20, activation='relu'))   
model.add(Dense(7, activation='softmax')) 


#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor="val_loss", patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.1, callbacks=[es])


#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0], 'accuracy: ', loss[1])

y_pred = model.predict(x_test)
print('accuracy_score:', accuracy_score(y_pred, y_test))

'''
epochs=100, batch_size=64, patience=10  :  [loss] 0.21259595453739166  [accuracy] 0.9155622720718384
'''