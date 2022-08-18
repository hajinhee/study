from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))

# vgg16.summary()
# vgg16.trainable = False     # 가중치를 동결시킨다!

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

# model.trainable = False    각 레이어하나마다 이름을 주고 동결시킬수도 있다.(Funtional)에서 더 편하게 쓸 수 있다.

# model.summary()
# print(len(model.weights))               # 각 layers = weights + bias가 1세트 2개.
# print(len(model.trainable_weights))     # layer가 1개 줄어들어서 len은 30이 나온다.