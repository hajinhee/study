from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Flatten, Conv1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  
import numpy as np

#1. 데이터로드 및 전처리 
docs = ['너무 재밌어요','참 최고에요','참 잘 만든 영화에요','추천하고 싶은 영화입니다','한 번 더 보고 싶네요','글세요',
        '별로에요','생각보다 지루해요','연기가 어색해요','재미없어요','너무 재미없다','참 재밌네요','예람이가 잘 생기긴 했어요']
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])    # 부정 0, 긍정 1


token = Tokenizer() # 토크나이저 선언
token.fit_on_texts(docs)
x = token.texts_to_sequences(docs)
pad_x = pad_sequences(x, padding='pre', maxlen=5)      
word_size = len(token.word_index)

#print(pad_x.shape) (13,5)

#2. 모델
model = Sequential()
model.add(Embedding(28,10,input_length=5))      # Dense하려면 input_length의 5값이 필요하기때문에 써야한다.
model.add(Conv1D(32, 2))
model.add(Flatten())
model.add(Dense(2,activation='softmax'))

#model.summary()

#3. 컴파일, 훈련

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=2)

#4. 평가, 예측
acc = model.evaluate(pad_x, labels,batch_size=1)[1]

print('acc : ',acc)

x_predict = ['반장이 재미없어요 어색해요 글쎄요 재미없다']    # 이거를 죽이되든 밥이되든 (1,5)로 만들어야한다.
#ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ [ ] 하나씌워주면 되는데 이걸 몰라서 20분을 고민하네.
token.fit_on_texts(x_predict)
x_predict = token.texts_to_sequences(x_predict)
y_pred = model.predict(x_predict)

#결과는 부정? 긍정?
부정 = round(y_pred[0][0]*100,2)
긍정 = round(y_pred[0][1]*100,2)

if y_pred[0][0] > y_pred[0][1]:
    print(f'{부정}% 의 확률로 부정적')
else : 
    print(f'{긍정}% 의 확률로 긍정적')

