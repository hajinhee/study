from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar100
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings

(x_train,y_train), (x_test,y_test) = cifar100.load_data()

input = Input(shape=(32,32,3))
vgg16 = VGG19(weights='imagenet',include_top=False)(input)
vgg16.trainable = True
Glp = GlobalAveragePooling2D()(vgg16)
hidden1 = Dense(1024)(Glp)
hidden2 = Dense(512)(hidden1)
output1 = Dense(100,activation='softmax')(hidden2)

model = Model(inputs=input, outputs=output1)

optimizer = Adam(learning_rate=1e-3)  # 1e-4     
lr=ReduceLROnPlateau(monitor= "val_acc", patience = 2, mode='max',factor = 0.1, min_lr=1e-6,verbose=False)
es = EarlyStopping(monitor ="val_acc", patience=5, mode='max',verbose=1,restore_best_weights=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics='acc')

start = time.time()
model.fit(x_train,y_train,batch_size=100,epochs=1000,validation_split=0.2,callbacks=[lr,es], verbose=True)
end = time.time()

loss, Acc = model.evaluate(x_test,y_test,batch_size=50,verbose=True)

print(f"Time : {round(end - start,4)}")
print(f"loss : {round(loss,4)}")
print(f"Acc : {round(Acc,4)}")

# 이게 Functional 방식이다.