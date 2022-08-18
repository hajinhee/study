from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout, Activation, MaxPooling2D


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', input_shape=(28, 28, 1)))  # ReLU를 활성화 함수로 사용하며 커널 수가 32인 합성곱층을 모델에 추가
              # 필터 수     # 필터의 크기                                       # 28*28 크기의 흑색조(1) 이미지, 컬러는 3
model.add(MaxPooling2D(pool_size=(2, 2)))  # 좋은 특징을 얻기 위해 이미지를 다운 샘플링
model.add(Conv2D(64, (3,3), strides=1, padding='same', activation='relu'))  #  커널 수를 64로 증가시킨 합성곱층
model.add(MaxPooling2D(pool_size=(2, 2)))  # 다시 한번 이미지 다운 샘플링
model.add(Flatten())  # 분류를 위해 추출된 특징을 1차원 벡터로 변환
model.add(Dense(64, activation='relu'))  # 변환된 특징벡터를 입력받는 전결합층
model.add(Dropout(rate=0.3))  # 30%의 노드를 비활성화하는 드롭아웃
model.add(Dense(5, activation='softmax'))  # softmax 함수로 클래스 10개의 확률을 출력하는 전결합층
model.summary()


'''
CNN 파라미터수(Param) 계산하는 방법
파라미터 수 = 이전 층 필터 수 * 이전 층 커널 크기 * 이전 층 출력의 깊이 + 현재 층 필터 수(편향)

 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        320        ----> (None, 특징맵 행 크기, 특징맵 열 크기, 필터 수) 

 max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0
 )

 conv2d_1 (Conv2D)           (None, 14, 14, 64)        18496     

 max_pooling2d_1 (MaxPooling  (None, 7, 7, 64)         0
 2D)

 flatten (Flatten)           (None, 3136)              0

 dense (Dense)               (None, 64)                200768    

 dense_1 (Dense)             (None, 5)                 325       

=================================================================
Total params: 219,909
Trainable params: 219,909
Non-trainable params: 0
'''