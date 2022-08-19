from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D,\
                                    GlobalAveragePooling2D,BatchNormalization,LayerNormalization
import numpy as np
from tensorflow.keras.datasets import cifar100  
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam,Adadelta
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler

#1.데이터 로드 및 정제
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float')/255
x_test = x_test.astype('float')/255
y_train = to_categorical(y_train)  # 원핫인코딩
y_test = to_categorical(y_test)
print(x_train.shape)

#2.모델링
model = Sequential()
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization()),model.add(MaxPooling2D())
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization()),model.add(MaxPooling2D())
model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(256,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization()),model.add(MaxPooling2D())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization()),model.add(MaxPooling2D())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization())
model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(LayerNormalization()),model.add(BatchNormalization()),model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1024,activation='relu')),model.add(LayerNormalization()),model.add(Dropout(0.5))
model.add(Dense(512,activation='relu')),model.add(LayerNormalization()),model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

#3. 모델 컴파일, 훈련, 저장
optimizer = Adam(learning_rate=0.0001)     
lr = ReduceLROnPlateau(monitor='val_acc', patience=2, mode='max', factor=0.1, min_lr=1e-6, verbose=False)
'''
[factor] 
Learning rate를 얼마나 감소시킬 지 정하는 인자값
새로운 learning rate는 기존 learning rate * factor
'''
es = EarlyStopping(monitor='val_acc', patience=3, mode='max', verbose=1, baseline=None, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras33_cifar100_MCP.hdf5')
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc']) 
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[lr, es])
model.save(f"./_save/keras33_save_cifar100.h5")

#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])


'''
[loss] :  3.0337114334106445
[accuracy] :  0.2630000114440918
'''