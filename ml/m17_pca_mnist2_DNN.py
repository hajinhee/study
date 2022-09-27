import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.utils import to_categorical
'''
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1) 
x_test = x_test.reshape(10000, 28,28,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

scaler = StandardScaler()

n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) 
x_train = scaler.fit_transform(x_train_reshape) 

x = np.append(x_train, x_test, axis=0)
x = x.reshape(70000,784)
print(x.shape)
pca = PCA(n_components=154)
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_
print(sum(pca_EVR))
print(pca_EVR)
cumsum = np.cumsum(pca_EVR)
print(cumsum)
print(np.argmax(cumsum > 0.95)+1)  # 712 > 0부터 시작하기 때문에 1을 더해야 한다.

# print(x_train.shape,x_test.shape)
# print(y_train.shape,y_test.shape)

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)) #.reshape(x_test.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_shape=(784, )))
# model.add(Dense(64, input_shape=(784, )))  # 위와 동일함
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(130, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Dense(10, activation = 'softmax'))


#3. 컴파일, 훈련
import time
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
start = time.time()
model.fit(x_train, y_train, epochs=16, batch_size=16,
          validation_split=0.3)
end = time.time()

#4. 평가, 예측
print ('====================== 1. 기본출력 ========================')
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2 스코어:', r2)
print('걸린 시간 :', end-start)


1. 나의 최고의 DNN
loss: [0.14787110686302185, 0.960099995136261]
r2 스코어: 0.9319709803598822
걸린 시간 : 208.04303908348083

2. 나의 최고의 CNN
time :
acc :

3. PCA 0.95
time :
acc :

4. PCA 0.99
time :
acc :

5. PCA 0.999
time :
acc :

6. PCA 1.0
time :
acc :
'''


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)  # x_train, x_test를 행으로 합친다는 뜻

scaler = StandardScaler()
x_train = x_train.reshape(60000, -1)  # (60000, 784)
x_test = x_test.reshape(10000, -1)  # (10000, 784)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# pca를 통해 0.95 이상인 n_components가 몇개?

pca = PCA(n_components=154)  # 칼럼이 28*28개의 벡터로 압축이됨
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print(x.shape)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)   
# print(sum(pca_EVR)) 

# cumsum = np.cumsum(pca_EVR)  
# print(cumsum[0])

# 2. 모델구성

model=Sequential()
model.add(Dense(64, input_shape=(154,)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)

import time
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2, callbacks=[es])
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

#4. 예측
loss = model.evaluate(x_test, y_test)
print("loss, accuracy : ", loss)

acc = str(round(loss[1], 4))
model.save("./_save/dnn_mnist_{}.h5".format(acc))
