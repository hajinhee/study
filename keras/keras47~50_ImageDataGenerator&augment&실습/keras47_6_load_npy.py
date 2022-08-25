from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,MaxPool2D,Conv2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

#1. 데이터 로드 및 전처리
x_train = np.load('../_data/_save_npy/keras47_5_train_x.npy')  # (160, 150, 150, 3)
y_train = np.load('../_data/_save_npy/keras47_5_train_y.npy')  # (160, )
x_test = np.load('../_data/_save_npy/keras47_5_test_x.npy')  # (120, 150, 150, 3)
y_test = np.load('../_data/_save_npy/keras47_5_test_y.npy')  # (120,)

#2. 모델링
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(150, 150, 3)))                                           
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 'sigmoid' -> 이진분류

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(monitor ='val_loss', patience=50, mode='min', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=1, batch_size=10, validation_split=0.2, callbacks=[es])  

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, batch_size=1)
print(' loss : ', round(loss[0], 4))
print(' acc : ', round(loss[1], 4))