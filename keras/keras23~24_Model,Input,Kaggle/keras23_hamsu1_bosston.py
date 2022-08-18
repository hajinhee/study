from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler  
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import numpy as np
import time
from sklearn.metrics import accuracy_score


#1.데이터 로드 및 정제
datasets = load_boston()
x = datasets.data  # (506, 13)
y = datasets.target  # (506,)
# print(x.shape)
# print(np.unique(y))  # 회귀문제
'''
[ 5.   5.6  6.3  7.   7.2  7.4  7.5  8.1  8.3  8.4  8.5  8.7  8.8  9.5
  9.6  9.7 10.2 10.4 10.5 10.8 10.9 11.  11.3 11.5 11.7 11.8 11.9 12.
 12.1 12.3 12.5 12.6 12.7 12.8 13.  13.1 13.2 13.3 13.4 13.5 13.6 13.8
 13.9 14.  14.1 14.2 14.3 14.4 14.5 14.6 14.8 14.9 15.  15.1 15.2 15.3
 15.4 15.6 15.7 16.  16.1 16.2 16.3 16.4 16.5 16.6 16.7 16.8 17.  17.1
..... ]
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49)

#scaler = MinMaxScaler()   
scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)  
x_test = scaler.transform(x_test)    


#2. 모델구성,모델링
input1 = Input(shape=(13))  # input shape
dense1 = Dense(40)(input1)  # input1에서 받아왔다
dense2 = Dense(30, activation='relu')(dense1)  # dense1에서 받아왔다
dense3 = Dense(20, activation='relu')(dense2)
dense3 = Dense(10, activation='relu')(dense2)
dense3 = Dense(5, activation='relu')(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)  # 함수형 모델은 inputs시작과 outputs 끝을 지정해준다.

# model = Sequential()
# model.add(Dense(50, input_dim=13))
# model.add(Dense(30))
# model.add(Dense(15,activation="relu")) #
# model.add(Dense(8,activation="relu")) #
# model.add(Dense(5))
# model.add(Dense(1))
# model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  # mse 평균제곱오차 = 회귀모델 loss함수
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
'''
[mode] 'auto' 또는 'min' 또는 'max'
monitor하는 값이 val_acc 즉 정확도일 경우, 값이 클수록 좋기때문에 'max'를 입력하고, val_loss일 경우 작을수록 좋기 때문에 'min'을 입력 'auto'는 모델이 알아서 판단한다.
[verbos] 0 또는 1
1인 경우 EarlyStopping이 적용될 때 화면에 적용되었다고 표시 후 종료되고 0인 경우 화면에 나타냄 없이 종료된다.
[baseline]
모델이 달성해야하는 최소한의 기준값. patience 이내에 모델이 baseline보다 개선됨이 보이지 않으면 Training을 중단시킨다.
예를 들어 patience가 3이고 baseline이 정확도기준 0.98 이라면, 3번의 trianing안에 0.98의 정확도를 달성하지 못하면 Training이 종료된다.
[restore_best_weights] True 또는 False
True라면 training이 끝난 후, model의 weight를 monitor하고 있던 값이 가장 좋았을 때의 weight로 복원한다. False라면, 마지막 training이 끝난 후의 weight를 유지한다.
'''

model.fit(x_train, y_train, epochs=500, batch_size=3, validation_split=0.111111, callbacks=[es])
ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")
print(krtime)

model.save(f"./_save/keras25_1_save_boston{krtime}.h5")

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)


'''
loss: 13.4399
r2스코어 :  0.8643167625510263
'''