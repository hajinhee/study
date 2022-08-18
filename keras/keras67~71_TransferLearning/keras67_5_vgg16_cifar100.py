from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time

(x_train,y_train), (x_test,y_test) = cifar100.load_data()

# print(x_train.shape,x_test.shape)               # 32,32,3
# print(len(np.unique(y_test)))                   # 100

x_train = x_train.reshape(50000,32,32,3)/255.
x_test = x_test.reshape(10000,32,32,3)/255.

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
# model = VGG16(weights=None, include_top=True, input_shape=(32,32,3), classes=100, pooling='max')

vgg16.trainable = False     # 가중치를 동결시킨다!

model = Sequential()
model.add(vgg16)
model.add(Flatten())
# model.add(GlobalAveragePooling2D())     
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(100,activation='softmax'))

optimizer = Adam(learning_rate=0.0001)      # 초기 lr이 되게 되게 중요하다
lr=ReduceLROnPlateau(monitor= "val_acc", patience = 2, mode='max',factor = 0.1, min_lr=0.00001,verbose=1)
es = EarlyStopping(monitor ="val_acc", patience=10, mode='max',verbose=1,restore_best_weights=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics='acc')

start = time.time()
model.fit(x_train,y_train,batch_size=100,epochs=10000,validation_split=0.2,callbacks=[lr,es])#,cp
end = time.time()
loss, Acc = model.evaluate(x_test,y_test,batch_size=100)

print(f"Time : {round(end - start,4)}")
print(f"loss : {round(loss,4)}")
print(f"Acc : {round(Acc,4)}")

# 결과 비교
# vgg trainable : True / False
# Flatten / Global Average Pooling
# 위 4개 조합해서 최고결과 뽑고 이전 최고치 acc0.65와 비교

# 출력결과     True/Flat    True/GAP      False/Flat      False/GAP     Defaultvgg16    
# time :       638.6476     280.0477      280.9244        536.9051          583.8867
# loss :       3.7577       2.1816        2.5506          2.5456            6.7649
#  acc :       0.6037       0.607         0.3655          0.3657            0.3121

# 수정완료