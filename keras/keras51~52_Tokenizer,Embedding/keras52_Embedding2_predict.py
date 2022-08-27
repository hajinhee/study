from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  
import numpy as np

#1. 데이터로드 및 전처리 
docs = ['너무 재밌어요', '참 최고예요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다', '한 번 더 보고싶네요', '글쎄요',
        '별로예요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요']
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])  # 부정 0, 긍정 1

token = Tokenizer() 
token.fit_on_texts(docs)  # 문장 리스트화
word_size = len(token.word_index)  # 빈도수순으로 단어에 인덱스 부여(빈도수가 많을수록 앞으로) --> 중복값X
x = token.texts_to_sequences(docs)  # 인덱스를 원래 문장 순서대로 나열 --> 중복O
'''
[[2, 3], [1, 4], [1, 5, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24]]
'''
pad_x = pad_sequences(x, padding='pre', maxlen=5)  # (12, 5)     

# 2. 모델링
model = Sequential()
'''
임베딩(embedding)은 변환한 벡터들이 위치한 공간이다. 
케라스에서 제공하는 도구인 Embedding()는 단어를 랜덤한 값을 가지는 밀집 벡터로 변환한 뒤에, 
인공 신경망의 가중치를 학습하는 것과 같은 방식으로 단어 벡터를 학습하는 방법을 사용한다.
'''
model.add(Embedding(28, 10))  
model.add(LSTM(32))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=2)

#4. 평가, 예측
acc = model.evaluate(pad_x, labels, batch_size=1)[1]  # [0]=loss, [1]=accuracy
print('acc : ', acc)  # acc :  1.0

x_test = ['영화가 정말 재밌어요 참 최고예요']  
token.fit_on_texts(x_test)  # 문장 리스트화
x_test = token.texts_to_sequences(x_test)  # [[3, 5, 1, 4, 6]] (1, 5)

y_pred = model.predict(x_test)  # [[0.00158436 0.9984156 ]]
print(y_pred)  

result = f'{round(y_pred[0][0]*100, 2)}% 의 확률로 부정적' if y_pred[0][0] > y_pred[0][1] else f'{round(y_pred[0][1]*100, 2)}% 의 확률로 긍정적'
print(result)

'''
99.84% 의 확률로 긍정적
'''