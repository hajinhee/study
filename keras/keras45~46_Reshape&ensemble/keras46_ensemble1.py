import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from icecream import ic
from tensorflow.keras.layers import Concatenate, concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

'''
앙상블 기법은 동일한 학습 알고리즘을 사용해 여러 모델을 학습하는 기법이다.
함수형 케라스를 활용해 앙상블 모델을 만들 수 있다.
각 함수형 케라스 모델을 리스트에 넣고 그를 Concatenate()하여 하나로 합쳐준다.
최종적으로 모델을 만들 땐 입력층 2개와 마지막 레이어인 outputs 부분만 파라미터로 넣어주면 된다.
ensemble은 행의 개수가 같아야 한다. 열의 개수는 달라도 되지만 개수(행은) 같아야 한다.
'''

#1. 데이터
x1 = np.array([range(100), range(301, 401)])  # (2, 100)
x2 = np.array([range(101, 201), range(411, 511), range(100,200)])  # (3, 100)

x1 = np.transpose(x1)  # (100, 2)
x2 = np.transpose(x2)  # (100, 3)

y = np.array(range(1001, 1101))  # (100, ) 

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.8, random_state=66)

#2. 모델구성
# model_1
input1 = Input(shape=(2))  # x1 데이터 입력
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(7, activation='relu', name='dense3')(dense2)
output1 = Dense(7, activation='relu', name='output1')(dense3) 

# model_2
input2 = Input(shape=(3))  # x2 데이터 입력
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13) 
output2 = Dense(5, activation='relu', name='output2')(dense14) 

# output1,2 merge
merge1 = Concatenate()([output1, output2])
merge2 = Dense(10, activation='relu')(merge1)
merge3 = Dense(7)(merge2)
last_output = Dense(1)(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)
model.summary()
'''
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_2 (InputLayer)           [(None, 3)]          0           []

 input_1 (InputLayer)           [(None, 2)]          0           []

 dense11 (Dense)                (None, 10)           40          ['input_2[0][0]']

 dense1 (Dense)                 (None, 5)            15          ['input_1[0][0]']

 dense12 (Dense)                (None, 10)           110         ['dense11[0][0]']

 dense2 (Dense)                 (None, 7)            42          ['dense1[0][0]']

 dense13 (Dense)                (None, 10)           110         ['dense12[0][0]']

 dense3 (Dense)                 (None, 7)            56          ['dense2[0][0]']

 dense14 (Dense)                (None, 10)           110         ['dense13[0][0]']

 output1 (Dense)                (None, 7)            56          ['dense3[0][0]']

 output2 (Dense)                (None, 5)            55          ['dense14[0][0]']

 concatenate (Concatenate)      (None, 12)           0           ['output1[0][0]',
                                                                  'output2[0][0]']

 dense (Dense)                  (None, 10)           130         ['concatenate[0][0]']

 dense_1 (Dense)                (None, 7)            77          ['dense[0][0]']

 dense_2 (Dense)                (None, 1)            8           ['dense_1[0][0]']

==================================================================================================
Total params: 809
Trainable params: 809
Non-trainable params: 0
__________________________________________________________________________________________________
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측 
loss = model.evaluate([x1_test, x2_test], y_test)
print('[loss]: ', loss[0])
print('[mae]: ', loss[1])

y_predict = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print('[r2_score]: ', round(r2, 4))

'''
[loss]:  0.0028534652665257454
[mae]:  0.011990356259047985
[r2_score]:  1.0
'''
