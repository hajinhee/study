from tensorflow.keras.datasets import imdb  # 리뷰 감성
import numpy as np
from tensorflow.python.keras.backend import binary_crossentropy
from icecream import ic
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

#1. 데이터 로드 및 정제
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
ic(x_train.shape, y_train.shape)  # (25000,), (25000,)  
ic(x_test.shape, y_test.shape)  # (25000,), (25000,)   
ic(np.unique(y_train, return_counts=True))  # [0, 1] -> 이진분류

ic('imdb 최대 길이:', max(len(i) for i in x_train))  # 2494
ic('imdb 평균 길이:', sum(map(len, x_train)) / len(x_train))  # 238.71364

x_train = pad_sequences(x_train, padding='pre', maxlen=240, truncating='pre')  # (25000, 240)
x_test = pad_sequences(x_test, padding='pre', maxlen=240, truncating='pre')  # (25000, 240)
 
# 원핫인코딩
y_train = to_categorical(y_train)  # (25000,) -> (25000, 2)
y_test = to_categorical(y_test)  # (25000,) -> (25000, 2)

#2. 모델링
model=Sequential()
model.add(Embedding(input_dim=10000, output_dim=10, input_length=240))
model.add(LSTM(32))
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(2, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print('acc:', acc)  #  0.8434399962425232
