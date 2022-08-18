from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import numpy as np
from pandas import get_dummies


#1.데이터 로드 및 정제
datasets = load_iris()
x = datasets.data  # (150, 4)
y = datasets.target   # (150,) 
y = get_dummies(y)  # 원핫인코딩 => 다중분류

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42) 

# #scaler = MinMaxScaler()   #어떤 스케일러 사용할건지 정의부터 해준다.
# #scaler = StandardScaler()
# #scaler = RobustScaler()
# scaler = MaxAbsScaler()
# scaler.fit(x_train)      
# x_train = scaler.transform(x_train)   
# x_test = scaler.transform(x_test)    

# #2. 모델구성,모델링

input1 = Input(shape=(4))
dense1 = Dense(70)(input1)
dense2 = Dense(55)(dense1)
dense3 = Dense(40,activation='relu')(dense2)
dense4 = Dense(25)(dense3)
dense5 = Dense(10,activation='relu')(dense4)
output1 = Dense(3,activation='softmax')(dense5)  # 다중분류 'softmax'
model = Model(inputs=input1, outputs=output1)

# model = Sequential()
# model.add(Dense(70, activation='linear', input_dim=4))    
# model.add(Dense(55))   
# model.add(Dense(40,activation='relu')) #
# model.add(Dense(25))
# model.add(Dense(10,activation='relu')) #
# model.add(Dense(3, activation='softmax'))  
# model.summary()

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 다중분류 'categorical_crossentropy'
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.11111111, callbacks=[es])

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0], 'accuracy: ', loss[1])

#5. 모델저장
model.save("./_save/keras25_4_save_iris.h5")
#model = load_model("./_save/keras25_4_save_iris.h5")


'''
[loss] :  0.09979528933763504  [accuracy]:  0.9333333373069763
'''