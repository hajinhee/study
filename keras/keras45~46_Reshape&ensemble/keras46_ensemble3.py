import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping

#1. 데이터
x1 = np.array([range(100), range(301, 401)])  # (2, 100)
x1 = np.transpose(x1)  # (100, 2)

y1 = np.array(range(1001, 1101))  # (100,)  
y2 = np.array(range(101, 201))  # (100,)  
y3 = np.array(range(401,501))  # (100,)  

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(x1, y1, y2, y3, train_size=0.8, random_state=66)

#2. 모델구성
# model_1
input1 = Input(shape=(2))
dense1 = Dense(5, activation='relu', name='dense1')(input1)
dense2 = Dense(7, activation='relu', name='dense2')(dense1)
dense3 = Dense(7, activation='relu', name='dense3')(dense2)
output1 = Dense(7, activation='relu', name='output1')(dense3) 

# output_model_1
output21 = Dense(7)(output1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)

# output_model_2
output31 = Dense(7)(output1)
output32 = Dense(21)(output31)
output33 = Dense(21)(output32)
output34 = Dense(11, activation='relu')(output33)
last_output2 = Dense(1)(output34)

# output_model_3
output41 = Dense(7)(output1)
output42 = Dense(21)(output41)
output43 = Dense(21)(output42)
output44 = Dense(11, activation='relu')(output43)
last_output3 = Dense(1)(output44)

model = Model(inputs=[input1], outputs=[last_output1, last_output2]) #,last_output3
model.summary()
'''
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to
==================================================================================================
 input_1 (InputLayer)           [(None, 2)]          0           []

 dense1 (Dense)                 (None, 5)            15          ['input_1[0][0]']

 dense2 (Dense)                 (None, 7)            42          ['dense1[0][0]']

 dense3 (Dense)                 (None, 7)            56          ['dense2[0][0]']

 output1 (Dense)                (None, 7)            56          ['dense3[0][0]']

 dense_4 (Dense)                (None, 7)            56          ['output1[0][0]']

 dense (Dense)                  (None, 7)            56          ['output1[0][0]']

 dense_5 (Dense)                (None, 21)           168         ['dense_4[0][0]']

 dense_1 (Dense)                (None, 11)           88          ['dense[0][0]']

 dense_6 (Dense)                (None, 21)           462         ['dense_5[0][0]']

 dense_2 (Dense)                (None, 11)           132         ['dense_1[0][0]']

 dense_7 (Dense)                (None, 11)           242         ['dense_6[0][0]']

 dense_3 (Dense)                (None, 1)            12          ['dense_2[0][0]']

 dense_8 (Dense)                (None, 1)            12          ['dense_7[0][0]']

==================================================================================================
Total params: 1,397
Trainable params: 1,397
Non-trainable params: 0
__________________________________________________________________________________________________
'''

#3. 컴파일, 훈련
model.compile(loss='mse',  optimizer='adam', metrics = 'mae')    
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1,baseline=None, restore_best_weights=True)
model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=1, validation_split=0.2, callbacks=[es]) 

#4.  평가, 예측 
loss = model.evaluate(x1_test, [y1_test, y2_test])   
print('[loss]: ', loss[0])
print('[mae]: ', loss[1])

y_predict = np.array(model.predict(x1_test)).reshape(2, 20)  # (2, 20)
r2_1 = r2_score(y1_test, y_predict[0])  # (20,) (2, 20)
r2_2 = r2_score(y2_test, y_predict[1])  # (20,) (2, 20)
print('[r2_1]: ', r2_1, '[r2_]: ', r2_2)    

'''
[loss]:  0.00014922358968760818
[mae]:  0.0001492224691901356
[r2_1]:  0.9999998111909641 
[r2_]:  0.9999999999985896
'''

