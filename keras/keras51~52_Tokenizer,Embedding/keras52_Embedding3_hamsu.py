from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  
import numpy as np
from icecream import ic

#1. 데이터로드 및 전처리 
docs = ['정말 재밌어요', '정말 최고의 영화입니다', '참 잘 만든 영화예요', '추천하고 싶은 영화입니다', '한번 더 보고싶네요', '글쎄요 완전 별로예요',
        '진짜 너무 별로예요', '생각보다 너무 지루해요', '연기가 너무 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요', '최악이에요']
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0])  # 부정 0, 긍정 1

token = Tokenizer() 
token.fit_on_texts(docs)
x = token.texts_to_sequences(docs)
pad_x = pad_sequences(x, padding='pre', maxlen=4)   # (13, 4)
ic(pad_x) 
'''
([[ 0,  0,  2,  6],
[ 0,  2,  7,  3],
[ 4,  8,  9, 10],
[ 0, 11, 12,  3],
[ 0, 13, 14, 15],
[ 0, 16, 17,  5],
[ 0, 18,  1,  5],
[ 0, 19,  1, 20],
[ 0, 21,  1, 22],
[ 0,  0,  0, 23],
[ 0,  0,  1, 24],
[ 0,  0,  4, 25],
[ 0,  0,  0, 26]]
'''  

#2. 모델링
input1 = Input(shape=(4,))  # 열
Embed = Embedding(28, 10)(input1)
LM = LSTM(32)(Embed)
dense = Dense(2, activation='softmax')(LM)
model = Model(inputs=input1, outputs=dense)

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=2)

#4. 평가, 예측
acc = model.evaluate(pad_x, labels, batch_size=1)[1]  # accuracy
print('[accuracy] : ', acc)  # [accuracy] :  1.0

x_test = ['영화가 너무 재미없어요 최악이에요']  # [[27, 1, 6, 7]]
token.fit_on_texts(x_test)
x_test = token.texts_to_sequences(x_test)  
y_pred = model.predict(x_test)

# 부정-0, 긍정-1
result = f'{round(y_pred[0][0]*100, 2)}% 의 확률로 부정적' if y_pred[0][0] > y_pred[0][1] else f'{round(y_pred[0][1]*100, 2)}% 의 확률로 긍정적'
print(result)

'''
94.47% 의 확률로 부정적
'''
