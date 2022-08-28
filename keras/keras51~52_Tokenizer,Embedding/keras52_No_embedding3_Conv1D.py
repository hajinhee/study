from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  
import numpy as np
from icecream import ic

#1. 데이터 로드 및 전처리 
docs = ['영화가 정말 최고예요', '정말 최고의 영화입니다', '정말 잘 만든 영화예요 재미있어요', '추천하고 싶은 영화입니다', '한번 더 보고싶네요', '글쎄요 완전 별로예요',
        '진짜 너무 별로예요', '생각보다 너무 지루해요', '연기가 너무 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요', '최악이에요']
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0])  # 부정 0, 긍정 1

token = Tokenizer()  # 토크나이저 선언
token.fit_on_texts(docs)  # 문장 리스트화
x = token.texts_to_sequences(docs)  # 빈도수순으로 인덱스가 부여된 단어를 원래 문장 순서대로 나열
pad_x = pad_sequences(x, padding='pre', maxlen=5)      
word_size = len(token.word_index)
ic(pad_x.shape)  # (13, 5)

# Conv1D 입력 위해 입력데이터 3차원 변환
pad_x = pad_x.reshape(pad_x.shape[0], 1, pad_x.shape[1])  # (13, 1, 5)

#2. 모델링
model = Sequential()
# model.add(Embedding(28, 10, input_length=5))      
model.add(Conv1D(10, 1, input_shape=(1, 5)))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=2)

#4. 평가, 예측
acc = model.evaluate(pad_x, labels, batch_size=1)[1]
print('acc : ', acc)  # acc :  0.9230769276618958

x_data = ['생각보다 너무 재미없어요 진짜 최악이에요']
token.fit_on_texts(x_data)  
x_data = token.texts_to_sequences(x_data) 
y_pred = model.predict([x_data])  
ic(type(y_pred))  # <class 'numpy.ndarray'>

result = f'{round(y_pred[0][0]*100, 2)}% 의 확률로 부정적' if y_pred[0][0] > y_pred[0][1] else f'{round(y_pred[0][1]*100, 2)}% 의 확률로 긍정적'
ic(result)