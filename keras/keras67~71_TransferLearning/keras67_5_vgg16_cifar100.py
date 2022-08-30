from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time
from icecream import ic

#1. load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
ic(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
ic(len(np.unique(y_test)))  # 100 

# normalize
x_train = x_train/255. 
x_test = x_test/255.

# load pretrained model 
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

vgg16.trainable = False # 가중치 고정

#2. modeling
model = Sequential()
model.add(vgg16)
# classifier 
model.add(GlobalAveragePooling2D())  # or model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(100, activation='softmax'))

#3. compile, train
optimizer = Adam(learning_rate=0.0001)     
lr=ReduceLROnPlateau(monitor='val_acc', patience=2, mode='max', factor=0.1, min_lr=0.00001, verbose=1)
es = EarlyStopping(monitor='val_acc', patience=10, mode='max', verbose=1, restore_best_weights=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics='acc')

start = time.time()
model.fit(x_train, y_train, batch_size=100, epochs=100, validation_split=0.2, callbacks=[lr, es])
end = time.time()

#4. evaluate
loss, acc = model.evaluate(x_test, y_test, batch_size=128)

print(f"Time : {round(end-start, 4)}")
print(f"loss : {round(loss, 4)}")
print(f"Acc : {round(acc, 4)}")

'''
Time : 395.3846
loss : 2.5466
Acc : 0.3636
'''