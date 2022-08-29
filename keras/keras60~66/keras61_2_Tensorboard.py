from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation
import numpy as np,time
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,TensorBoard
from tensorflow.keras.optimizers import Adam
from icecream import ic

#1. load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
ic(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,)

# reshape
x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1)

#2. modeling
model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3), input_shape=(28,28,1)))  
model.add(Conv2D(10, (3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10, (2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. compile, train
learning_rate = 0.01           
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc']) 
es = EarlyStopping(monitor='val_acc', patience=15, mode='max', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, mode='max', verbose=1, min_lr=0.0001, factor=0.5) 
tb = TensorBoard(log_dir='./_data/_save_dat/_graph', histogram_freq=0, write_graph=True, write_images=True)
'''
TensorBoard(): 이 콜백은 다음을 포함하여 TensorBoard에 대한 이벤트를 기록한다.
-메트릭 요약 플롯
-훈련 그래프 시각화
-가중치 히스토그램
-샘플링된 프로파일링
log_dir:        TensorBoard에서 구문 분석할 로그 파일을 저장할 디렉토리의 경로
histogram_freq:	histogram_freq=1 으로 설정하면 모든 에포크마다 히스토그램 계산을 활성화 -> dafault=0
write_graph:    TensorBoard에서 그래프를 시각화할지 여부 -> write_graph가 True로 설정되면 로그 파일이 상당히 커질 수 있음
write_images:	TensorBoard에서 이미지로 시각화하기 위해 모델 가중치를 작성할지 여부
'''
start = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=50, validation_split=0.2, callbacks=[reduce_lr, es, tb])
end = time.time()-start

#4. evaluate
loss, acc = model.evaluate(x_test, y_test)
ic(loss, acc)
'''
loss: 0.22606267035007477, acc: 0.9325000047683716
'''
ic(learning_rate, round(loss, 4), round(acc, 4), f'걸린 시간: {round(end, 4)}')

'''
learning_rate: 0.01
round(loss, 4): 0.2261
round(acc, 4): 0.9325
f'걸린 시간: {round(end, 4)}': '걸린 시간: 216.1567'
'''


####################### visualization ###########################

# TensorBoard에서 저장 후 cmd에서 해당 경로 들어가서 실행하면 인터넷창에서 결과를 볼 수 있다.

# Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
# TensorBoard 2.7.0 at http://localhost:6006/ (Press CTRL+C to quit)