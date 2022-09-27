import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron   ## 원조
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터   ### xor 데이터 ### 
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
# model = Perceptron()     # xor에 막혔다. 
# model = SVC()
model = Sequential()
model.add(Dense(64,input_dim = 2,activation = 'relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1,activation= 'sigmoid'))
# model.summary()

#3. 훈련
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size = 1, epochs = 100)

#4. 평가, 예측
y_pred = model.predict(x_data)
print(x_data, " 의 예측결과 : ", y_pred)
results = model.evaluate(x_data, y_data)
print('metrics_acc : ', results[1])


# acc = accuracy_score(y_data, y_pred)
# print('accuracy_score : ', acc)