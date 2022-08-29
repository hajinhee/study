import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from icecream import ic

#1. load data
x = np.array([1, 2, 3, 4, 5])  # (5, )
y = np.array([1, 2, 3, 4, 5])  # (5, )

#2. modeling
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))
# model.summary()
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 17
Trainable params: 17
Non-trainable params: 0
_________________________________________________________________
'''

ic(model.weights)
'''
<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.11560118,  1.0606455 , -1.0812484 ]], dtype=float32)>, 
<tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>

<tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=array([[ 0.03904426,  0.3300786 ],
[-0.97085786, -0.14864534],[ 0.6138109 , -0.98611933]], dtype=float32)>,
<tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>

<tf.Variable 'dense_2/kernel:0' shape=(2, 1) dtype=float32, numpy=array([[ 0.9262482],[-0.4671206]], dtype=float32)>, 
<tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
'''

ic(model.trainable_weights)

print(len(model.weights))
print(len(model.trainable_weights))  # 17

model.trainable=False  # 가중치 갱신을 하지 않겠다 -> 전이학습 시 최적의 가중치로 시작하기 때문에 유용하다.

print(len(model.weights))
print(len(model.trainable_weights))  # 0

# model.summary() 
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 3)                 6

 dense_1 (Dense)             (None, 2)                 8

 dense_2 (Dense)             (None, 1)                 3

=================================================================
Total params: 17
Trainable params: 0
Non-trainable params: 17
_________________________________________________________________
'''

#3. compile, train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, batch_size=1, epochs=100)  # model.trainable=False -> loss 값이 갱신되지 않는다.