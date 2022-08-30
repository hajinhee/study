from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. modeling
model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.layers[0].trainable = False  # Dense
model.layers[1].trainable = False  # Dense_1
model.layers[2].trainable = False  # Dense_2

# model.trainable = False
for layer in model.layers:
    layer.trainable = False

model.summary()
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

print(model.layers)
'''
[<keras.layers.core.dense.Dense object at 0x000001DB641A4760>, 
<keras.layers.core.dense.Dense object at 0x000001DB2DEA1040>, 
<keras.layers.core.dense.Dense object at 0x000001DB2DE8D880>]     
'''