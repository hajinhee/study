from tensorflow.keras.models import Sequential, Model,load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from pandas import get_dummies

#1.데이터 로드 및 정제
datasets = fetch_covtype()
x = datasets.data  # (581012, 54)
y = datasets.target  # (581012,)      
# print(x.shape)  
# print(y.shape)
# print(np.unique(y)) -> [1 2 3 4 5 6 7]
y = get_dummies(y)  # 원핫인코딩 -> 다중분류

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49) 


#scaler = MinMaxScaler()   #어떤 스케일러 사용할건지 정의부터 해준다.
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
#scaler.fit(x_train)       #어떤 비율로 변환할지 계산해줌.
#x_train = scaler.transform(x_train)   
#x_test = scaler.transform(x_test)    


#2. 모델구성,모델링
input1 = Input(shape=(54))
dense1 = Dense(100,activation='relu')(input1)
dense2 = Dense(80)(dense1)
dense3 = Dense(60,activation='relu')(dense2)
dense4 = Dense(40)(dense3)
dense5 = Dense(20)(dense4)
output1 = Dense(7,activation='softmax')(dense5)
model = Model(inputs=input1,outputs=output1)

# model = Sequential()
# model.add(Dense(100, activation='linear', input_dim=54))    
# model.add(Dense(80))
# model.add(Dense(60 ,activation="relu")) #
# model.add(Dense(40))
# model.add(Dense(20 ,activation="relu")) #  
# model.add(Dense(7, activation='softmax')) 


#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=100, batch_size=1000, validation_split=0.11111111, callbacks=[es])

#4. 모델저장
model.save("./_save/keras25_6_save_covtype.h5")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0], 'accuracy: ', loss[1])

'''
[loss]:  0.4046468138694763  [accuracy]:  0.8320884108543396
'''