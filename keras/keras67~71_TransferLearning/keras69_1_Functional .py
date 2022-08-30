from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar100
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings
from icecream import ic

#1. load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
ic(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

#2. modeling
input = Input(shape=(32, 32, 3))
vgg16 = VGG19(weights='imagenet', include_top=False)(input)
vgg16.trainable = True
# classifier 
glp = GlobalAveragePooling2D()(vgg16)
hidden1 = Dense(1024)(glp)
hidden2 = Dense(512)(hidden1)
output1 = Dense(100, activation='softmax')(hidden2)
model = Model(inputs=input, outputs=output1)

#3. compile, train
optimizer = Adam(learning_rate=1e-3)  # 1e-4     
lr=ReduceLROnPlateau(monitor='val_acc', patience=2, mode='max', factor=0.1, min_lr=1e-6, verbose=False)
es = EarlyStopping(monitor='val_acc', patience=5, mode='max', verbose=1, restore_best_weights=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics='acc')

start = time.time()
model.fit(x_train, y_train, batch_size=100, epochs=100, validation_split=0.2, callbacks=[lr, es], verbose=True)
end = time.time()

#4. evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size=128, verbose=True)

print(f"Time : {round(end-start, 4)}")
print(f"loss : {round(loss, 4)}")
print(f"Acc : {round(acc, 4)}")

'''
Time : 180.8773
loss : 4.6063
Acc : 0.01
'''