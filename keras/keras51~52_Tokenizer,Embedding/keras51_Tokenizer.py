from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 매우 jonna 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])  # 문장을 띄어쓰기 구분해서 끊어서 중복없이 인덱스 번호를 매겨준다. 단 앞에서부터 순서대로 인덱스번호가 들어가진 않는다.

#print(token.word_index)     # 작업된 값에서 단어의 인덱스 번호를 출력해본다.
#{'진짜': 1, '마구': 2, '나는': 3, '매우': 4, 'jonna': 5, '맛있는': 6, '밥을': 7, '먹었다': 8}

x = token.texts_to_sequences([text])    # 문장의 원래 순서(인덱스번호)대로 저장한 후 출력해보겠다. fit했기때문에 값이 생긴것.
#print(x)                                # 인덱스 번호는 int형이므로 각 어절에 value값이 생겨서 원핫인코딩 해줘야한다.
#[[3, 1, 4, 5, 6, 7, 1, 2, 2, 8]]

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
#print(word_size)        # 8

x = to_categorical(x)
print(x)
print(x.shape)
'''
[[[0. 0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 1. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 1. 0.]
  [0. 1. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
(1, 10, 9)                          10은 어절이 10개라서 (중복되버렸네?) 카테코리컬이라 맨앞에 0이 들어가서 + 1해서 9 되고 
'''
