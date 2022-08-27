from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

#1. 데이터
docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요', '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', '너무 재미없다', '참 재밌네요']

# 긍정 1, 부정 0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])    # 11개의 문장이 긍정인지 부정인지 내가 판단해서 라벨값을 넣어 1개의 배열로 만들어준다.

token = Tokenizer()
token.fit_on_texts(docs)  # 문장을 리스트화
# print(token.word_index)  # 빈도수대로 인덱스가 부여(빈도수가 높으면 앞으로)된 딕셔너리 형태의 단어
'''
{'참': 1, '너무': 2, '잘': 3, '재밌어요': 4, '최고에요': 5, '만든': 6, '영화에요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10,
'한': 11, '번': 12, '더': 13, '보고': 14, '싶네요': 15, '글세요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20,
'어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밌네요': 24, '예람이가': 25, '생기긴': 26, '했어요': 27}   
'''

x = token.texts_to_sequences(docs)  # 원래 문장 순서대로 인덱스 나열 
# print(x)
'''
[[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17], [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]
'''

pad_x = pad_sequences(x, padding='pre', maxlen=5)  # 가장 긴 문장의 길이인 '5'가 maxlen이 된다.  # 요소를 패드로 채워준다. 앞에 0을 넣어줄수도 뒤에 0을 넣어줄수도 있지만 거의 앞을채워줌
'''
자연어 처리를 하다보면 각 문장(또는 문서)은 서로 길이가 다를 수 있다. 
그런데 기계는 길이가 전부 동일한 문서들에 대해서 하나의 행렬로 보고, 한꺼번에 묶어서 처리할 수 있다.
다시 말해 병렬 연산을 위해서 여러 문장의 길이를 임의로 동일하게 맞춰주는 작업이 필요할 때가 있다. 
가장 길이가 긴 문장의 길이에 맞춰 모든 문장의 길이를 맞춰준다. 이때 가상의 단어 'PAD'를 사용한다.
'PAD'라는 단어가 있다고 가정하고, 이 단어는 0번 단어라고 정의한다. 길이가 기준(가장 긴 문장)보다 짧은 문장에는 숫자 0을 채워서 길이를 맞춰준다.
이와 같이 데이터에 특정 값을 채워서 데이터의 크기(shape)를 조정하는 것을 패딩(padding)이라고 한다. 숫자 0을 사용하고 있다면 제로 패딩(zero padding)이라고 한다.
'''



print(pad_x)    
'''
[[ 0  0  0  2  4]
 [ 0  0  0  1  5]
 [ 0  1  3  6  7]
 [ 0  0  8  9 10]
 [11 12 13 14 15]
 [ 0  0  0  0 16]
 [ 0  0  0  0 17]
 [ 0  0  0 18 19]
 [ 0  0  0 20 21]
 [ 0  0  0  0 22]
 [ 0  0  0  2 23]
 [ 0  0  0  1 24]
 [ 0 25  3 26 27]]      (13, 5) <class 'numpy.ndarray'> 타입까지 바뀌어서 이제 모델링이 가능해졌다.
'''

word_size = len(token.word_index)
print("word_size : ", word_size)    # 0이 추가되기전 27개.
print(np.unique(pad_x))             # 0이 추가되어 28개!
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27]

#여기서 이제 원핫인코딩하면 어...? labels값이 28개네? (13,5) -> (13,5,28) ㄷㄷ  -> Embedding
# Embedding 해주는거 다른곳에서도 쓸수있나?



#2. 모델
model = Sequential()
#                                                인풋은 (13, 5)
#                 단어의 index+0값 아웃풋출력개수(노드개수)  단어수, 길이
#model.add(Embedding(input_dim=28,   output_dim=10,      input_length=5))        # 5는 최대 어절의 길이
#model.add(Embedding(28,10,input_length=5))
model.add(Embedding(28,10))    #output_dim input_dim            outputdim은 단어의 개수27 + pad_sequences 0해서 28개!
# 임배딩작업은 인풋x값의 백터화. (13,5,28) 원핫인코딩 했다면 1차원이므로, 하지만 벡터화하면 2차원으로 들어가서 파라미터연산 횟수보면 28 * 10으로 들어간다.
# 임배딩은 3차원 아웃풋이므로 통상 LSTM(시계열)으로 받는다. 
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================       
 embedding (Embedding)       (None, 5, 10)             280              # 28 * 10 = 280    파라미터 연산횟수가 280

 embedding (Embedding)       (None, None, 10)          280              # input_length=5는 굳이명시안해줘도 위에서 이미 손질 다 해놨기때문에 알아서 돌아감.
'''
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.summary()

'''
#3. 컴파일, 훈련

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=32)

#4. 평가, 예측
acc = model.evaluate(pad_x, labels)[1]

print('acc : ',acc)
'''