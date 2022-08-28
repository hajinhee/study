from tensorflow.keras.datasets import reuters  # 로이터 뉴스 분류
import numpy as np 
import operator
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Flatten
from icecream import ic
from tensorflow.keras.preprocessing.text import Tokenizer

#1. 데이터 로드 및 정제
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)
ic(x_train.shape, y_train.shape)  # (8982,), (8982,)
ic(x_test.shape, y_test.shape)  #  (2246,), (2246,)
ic(np.unique(y_train, return_counts=True))  # 다중분류 
'''
[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
'''
ic(type(x_train[0]), len(x_train[0]))  # <class 'list'>,  87
ic(type(x_train[1]), len(x_train[1]))  # <class 'list'>,  56

ic('뉴스 기사의 최대 길이:', max(len(i) for i in x_train))   # 2376
ic('뉴스 기사의 최대 길이:', max(len(i) for i in x_test))   # 1032
ic('뉴스 기사의 평균 길이:', sum(map(len, x_train)) / len(x_train))  # 145.5398574927633

x_train = pad_sequences(x_train, padding='pre', maxlen=1032, truncating='pre')  
x_test = pad_sequences(x_test, padding='pre', maxlen=1032, truncating='pre')
'''
padding: post-padding보다는 pre-padding이 성능이 좋고 많이 사용된다.(default=pre)
truncating: maxlen보다 더 긴 문장이 들어왔을 때 해당 문장을 maxlen에 맞춰서 truncating='pre'이면 앞에서부터, truncating='post'면 뒤에서부터 단어를 자른다.
'''
ic(x_train.shape, y_train.shape)  # (8982, 2376), (8982,) --> 최대길이에 맞춰서 쉐입이 변한다.
ic(x_test.shape, y_test.shape)  # (2246, 1032), (2246,)

# 원핫인코딩
y_train = to_categorical(y_train)  # (8982,) -> (8982, 46)
y_test = to_categorical(y_test)  # (2246,) -> (2246, 46)

#2. 모델링
model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=10, input_length=1032)) 
model.add(LSTM(32))
model.add(Dense(64, activation='relu'))
model.add(Dense(32))                                                  
model.add(Dense(16))                                                                                                    
model.add(Dense(46, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4. 평가, 예측
acc = model.evaluate(x_test, y_test)[1]
print('acc:', acc)  # acc: 0.5690115690231323
