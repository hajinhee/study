from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler  # 미리 처리한다 -> 전처리
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import numpy as np


#1.데이터 로드 및 정제
datasets = load_boston()
x = datasets.data  # (506, 13)
y = datasets.target  # (506,)

print(y)  # 분류, 회귀모델 확인 -> 회귀모델
'''
[24.  21.6 34.7 33.4 36.2 28.7 22.9 27.1 16.5 18.9 15.  18.9 21.7 20.4
 18.2 19.9 23.1 17.5 20.2 18.2 13.6 19.6 15.2 14.5 15.6 13.9 16.6 14.8
 18.4 21.  12.7 14.5 13.2 13.1 13.5 18.9 20.  21.  24.7 30.8 34.9 26.6
 25.3 24.7 21.2 19.3 20.  16.6 14.4 19.4 19.7 20.5 25.  23.4 18.9 35.4
 24.7 31.6 23.3 19.6 18.7 16.  22.2 25.  33.  23.5 19.4 22.  17.4 20.9
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)   

##################################### 스케일러 설정 옵션 ########################################
'''
원본 데이터는 데이터 고유의 특성과 분포를 가지고 있다.
이를 그대로 사용하게 되면 학습이 느리거나 문제가 발생하는 경우가 종종 발생하여
Scaler를 이용하여 동일하게 일정 범위로 스케일링하는 것이 필요하다.

[StandardScaler]
평균을 제거하고 데이터를 단위 분산으로 조정한다. 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다.
따라서 이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없다.

***[MinMaxScaler] 데이터 편차가 클 때 사용
모든 feature값이 0~1 사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있다.
즉, MinMaxScaler 역시 아웃라이어의 존재에 매우 민감하다.

[MaxAbsScaler]
절대값이 0~1사이에 매핑되도록 한다. 즉 -1~1 사이로 재조정한다. 양수 데이터로만 구성된 특징 데이터셋에서는 MinMaxScaler와 유사하게 동작하며, 큰 이상치에 민감할 수 있다.

***[RobustScaler] 데이터 편차가 크지 않을 때 사용
아웃라이어의 영향을 최소화한 기법이다. 중앙값(median)과 IQR(interquartile range)을 사용하기 때문에 StandardScaler와 비교해보면 표준화 후 동일한 값을 더 넓게 분포 시키고 있음을 확인 할 수 있다.
IQR = Q3 - Q1 : 즉, 25퍼센타일과 75퍼센타일의 값들을 다룬다.
'''
scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)    
x_test = scaler.transform(x_test)    

#2. 모델구성,모델링
model = Sequential()
model.add(Dense(50, input_dim=13))
model.add(Dense(30))
model.add(Dense(15, activation='relu')) 
model.add(Dense(8, activation='relu')) 
model.add(Dense(5))
model.add(Dense(1))  # 회귀모델: linear(default값), 이진분류: sigmoid, 다중분류: softmax
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 50)                700
_________________________________________________________________
dense_1 (Dense)              (None, 30)                1530
_________________________________________________________________
dense_2 (Dense)              (None, 15)                465
_________________________________________________________________
dense_3 (Dense)              (None, 8)                 128
_________________________________________________________________
dense_4 (Dense)              (None, 5)                 45
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 6
=================================================================
Total params: 2,874
Trainable params: 2,874
Non-trainable params: 0
_________________________________________________________________
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  # 회귀모델: mse, 이진분류: binary_crossentropy, 다중분류: categorical_crossentropy

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=500, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)


# epochs=500, batch=1, patience=50  [loss] :  11.5029  [r2스코어] :  0.843143306629888 
