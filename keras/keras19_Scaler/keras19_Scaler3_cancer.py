from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.utils import to_categorical 
from icecream import ic 
from sklearn.preprocessing import OneHotEncoder            

#1.데이터 로드 및 정제
datasets = load_breast_cancer()
x = datasets.data # (569, 30)
y = datasets.target  # (569,)
print(np.unique(y))  # [0 1]  2개의 값 -> 이진분류 one hot encoding

# y = to_categorical(y) # (569, 2)
en = OneHotEncoder(sparse=False)         
y = en.fit_transform(y.reshape(-1, 1))     
ic(y.shape)  # (569, 2)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42) 

scaler = RobustScaler()  # 일정 범위로 스케일링
x_train = scaler.fit_transform(x_train)    
x_test = scaler.transform(x_test) 

#2. 모델구성,모델링
model = Sequential()
model.add(Dense(50, input_dim=30))  # 입력데이터 x의 열 값을 넣어준다.   
model.add(Dense(40 ,activation='relu')) 
model.add(Dense(30 ,activation='relu')) 
model.add(Dense(20))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(2, activation='sigmoid'))  # 이진분류 모델 출력 활성화함수
model.summary()

#3. 컴파일 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor="val_loss", patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=10000, batch_size=10, validation_split=0.1, callbacks=[es])

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0], 'accuracy: ', loss[1])


#  MinMaxScaler, epochs=10000, batch_size=5,  [loss]:  0.5812874436378479  [accuracy]:  0.9473684430122375

#  layer, node 추가, 스케일러와 배치사이즈 변경 후
#  RobustScaler, epochs=10000, batch_size=10,  [loss]:  0.1645577847957611  [accuracy]:  0.9824561476707458