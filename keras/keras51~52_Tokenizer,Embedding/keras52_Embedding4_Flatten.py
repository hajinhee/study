from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  
import numpy as np
from icecream import ic

#1. 데이터로드 및 전처리 
docs = ['영화가 정말 최고예요', '정말 최고의 영화입니다', '정말 잘 만든 영화예요 재미있어요', '추천하고 싶은 영화입니다', '한번 더 보고싶네요', '글쎄요 완전 별로예요',
        '진짜 너무 별로예요', '생각보다 너무 지루해요', '연기가 너무 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요', '최악이에요']
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0])  # 부정 0, 긍정 1

token = Tokenizer() 
token.fit_on_texts(docs)
word_size = len(token.word_index)  # 26
'''
'글쎄요': 16,    
'너무': 1,       
'더': 14,        
'만든': 9,       
'별로예요': 5,   
'보고싶네요': 15,
'생각보다': 19,  
'싶은': 12,      
'어색해요': 22,  
'연기가': 21,
'영화예요': 10,
'영화입니다': 3,
'완전': 17,
'잘': 8,
'재미없다': 24,
'재미없어요': 23,
'재밌네요': 25,
'정말': 2,
'지루해요': 20,
'진짜': 18,
'참': 4,
'최고예요': 6,
'최고의': 7,
'최악이에요': 26,
'추천하고': 11,
'한번': 13}
'''
x = token.texts_to_sequences(docs)  # 13
'''
[[2, 6],
[2, 7, 3],
[4, 8, 9, 10],
[11, 12, 3],
[13, 14, 15],
[16, 17, 5],
[18, 1, 5],
[19, 1, 20],
[21, 1, 22],
[23],
[1, 24],
[4, 25],
[26]]
'''
pad_x = pad_sequences(x, padding='pre', maxlen=5)   # (13, 5)
ic(pad_x) 
'''
[[ 0,  0,  5,  2,  6],
[ 0,  0,  2,  7,  3],
[ 2,  8,  9, 10, 11],
[ 0,  0, 12, 13,  3],
[ 0,  0, 14, 15, 16],
[ 0,  0, 17, 18,  4],
[ 0,  0, 19,  1,  4],
[ 0,  0, 20,  1, 21],
[ 0,  0, 22,  1, 23],
[ 0,  0,  0,  0, 24],
[ 0,  0,  0,  1, 25],
[ 0,  0,  0, 26, 27],
[ 0,  0,  0,  0, 28]]
'''  

#2. 모델링
model = Sequential()
model.add(Embedding(28, 10, input_length=5))  #  열
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
# model.summary()
'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, 5, 10)             280

 flatten (Flatten)           (None, 50)                0

 dense (Dense)               (None, 2)                 102

=================================================================
Total params: 382
Trainable params: 382
Non-trainable params: 0
_________________________________________________________________
'''

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=2)

#4. 평가, 예측
acc = model.evaluate(pad_x, labels, batch_size=1)[1]  # [0]=loss, [1]=accuracy
print('acc : ', acc)  # acc :  1.0

x_test = ['최악이에요 너무 재미없어 너무 지루해요']  # [[6, 1, 29, 1, 5]]
token.fit_on_texts(x_test)  
x_test = token.texts_to_sequences(x_test)  
ic(x_test)
y_pred = model.predict(x_test) 
ic(y_pred)  # [[0.6543308 , 0.34566918]]

# 부정-0, 긍정-1
result = f'{round(y_pred[0][0]*100, 2)}% 의 확률로 부정적' if y_pred[0][0] > y_pred[0][1] else f'{round(y_pred[0][1]*100, 2)}% 의 확률로 긍정적'
print(result)

'''
65.43% 의 확률로 부정적
'''