from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from pandas import get_dummies

#1.데이터 로드 및 정제
datasets = load_wine()
x = datasets.data  # (178, 13)
y = datasets.target  # (178,) 
# print(x.shape)
# print(y.shape)
# print(np.unique(y))  -> [0 1 2]
y = get_dummies(y)  # 원핫인코딩 다중분류 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42) 

#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)   
x_test = scaler.transform(x_test)    


#2. 모델구성,모델링
input1 = Input(shape=(13))
dense1 = Dense(120)(input1)
dense2 = Dense(100, activation='relu')(dense1)
dense3 = Dense(80)(dense2)
dense4 = Dense(60,activation='relu')(dense3)
dense5 = Dense(40)(dense4)
dense6 = Dense(20)(dense5)
output1 = Dense(3,activation='softmax')(dense6)
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(120, activation='linear', input_dim=13))    
# model.add(Dense(100 ,activation='relu')) #  
# model.add(Dense(80))
# model.add(Dense(60 ,activation='relu'))  # 
# model.add(Dense(40))
# model.add(Dense(20))
# model.add(Dense(3, activation='softmax'))
#model.summary()


#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.1111111111, callbacks=[es])

#4. 모델저장
model.save("./_save/keras25_5_save_wine.h5")

#5. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0], 'accuracy: ', loss[1])

'''
[loss] :  0.058120861649513245   [accuracy]:  1.0
'''