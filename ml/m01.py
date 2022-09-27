import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


#1. 데이터

datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(150, 4) (150,)
# print(y)
# print(np.unique(y))
# from sklearn.preprocessing import OneHotEncoder
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape) # (150,3)


x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 46)

# print(x_train.shape, y_train.shape)  # (120, 4) (120, 3)
# print(x_test.shape, y_test.shape)   # (30, 4) (30, 3)


#2. 모델구성
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC

# model = Sequential()
# model.add(Dense(50, activation= 'linear', input_dim=4))
# model.add(Dense(70, activation= 'sigmoid')) 
# model.add(Dense(50, activation= 'linear'))
# model.add(Dense(50, activation= 'linear'))
# model.add(Dense(50, activation= 'sigmoid'))
# model.add(Dense(3, activation= 'softmax')) 

model = LinearSVC()      # 단층으로만 구성됨

#3. 컴파일, 훈련
# model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
 
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience=10, mode='auto',
#                    verbose=1, restore_best_weights=False) # restore_best_weights=True : 최종값 이전에 가장 좋았던 값 도출함

# model.fit(x_train, y_train, epochs=100, batch_size=5,
#           validation_split=0.3)

model.fit(x_train,y_train)

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
# print('loss :', loss[0])
# print('accuracy :', loss[1])
result = model.score(x_test, y_test)

# 머신러닝은 분류와 회귀 모델 밖에 없다.

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('result:', result)
print('accuracy_score :', acc)

######## 원 핫 인 코딩 하지 않아도 된다. 