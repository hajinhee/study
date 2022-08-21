from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Concatenate, concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score

#1. 데이터
x1 = np.array([range(100), range(301, 401)])  # (2, 100)
x2 = np.array([range(101, 201), range(411, 511), range(100,200)])  # (3, 100)

x1 = np.transpose(x1)  # (100, 2)
x2 = np.transpose(x2)  # (100, 3)

y1 = np.array(range(1001, 1101))  # (100,)
y2 = np.array(range(101, 201))  # (100,)

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1, x2, y1, y2, train_size=0.8, random_state=66)
print(x1_train.shape, x1_test.shape)  # (80, 2) (20, 2)
print(x2_train.shape, x2_test.shape)  # (80, 3) (20, 3)
print(y1_train.shape, y1_test.shape)  # (80,) (20,)
print(y2_train.shape, y2_test.shape)  # (80,) (20,)

#2. 모델구성
# model_1
input1 = Input(shape=(2))  # x1
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(7, activation='relu', name='dense3')(dense2)
output1 = Dense(7, activation='relu', name='output1')(dense3) 

# model_2
input2 = Input(shape=(3))  # x2
dense11 = Dense(10, activation='relu', name='dense11')(input2)
dense12 = Dense(10, activation='relu', name='dense12')(dense11)
dense13 = Dense(10, activation='relu', name='dense13')(dense12)
dense14 = Dense(10, activation='relu', name='dense14')(dense13) 
output2 = Dense(5, activation='relu', name='output2')(dense14) 

merge1 = Concatenate()([output1, output2])

# output_model_1
output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)

# output_model_2
output31 = Dense(7)(merge1)
output32 = Dense(21)(output31)
output33 = Dense(21)(output32)
output34 = Dense(11, activation='relu')(output33)
last_output2 = Dense(1)(output34)

model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])
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

 dense_4 (Dense)                (None, 7)            91          ['concatenate[0][0]']

 dense (Dense)                  (None, 7)            91          ['concatenate[0][0]']

 dense_5 (Dense)                (None, 21)           168         ['dense_4[0][0]']

 dense_1 (Dense)                (None, 11)           88          ['dense[0][0]']

 dense_6 (Dense)                (None, 21)           462         ['dense_5[0][0]']

 dense_2 (Dense)                (None, 11)           132         ['dense_1[0][0]']

 dense_7 (Dense)                (None, 11)           242         ['dense_6[0][0]']

 dense_3 (Dense)                (None, 1)            12          ['dense_2[0][0]']

 dense_8 (Dense)                (None, 1)            12          ['dense_7[0][0]']

==================================================================================================
Total params: 1,892
Trainable params: 1,892
Non-trainable params: 0
__________________________________________________________________________________________________
'''

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics='mae')  
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, baseline=None, restore_best_weights=True)
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측 
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test])
y_predict = model.predict([x1_test, x2_test])
print('[loss]: ', loss[0])
print('[mae]: ', loss[1])

'''
[loss]:  0.026815930381417274
[mae]:  0.02620873786509037
'''