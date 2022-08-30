from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
'''
사전 학습된 가중치를 내려받아 변수 vgg16 에 할당한다. 
이때 'imagenet' 데이터셋으로 학습된 가중치를 지정하고, include_top=False 로 지정해서 분류기 부분의 가중치는 내려받지 않는다. 
'''
vgg16.summary()
'''
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 32, 32, 3)]       0

 block1_conv1 (Conv2D)       (None, 32, 32, 64)        1792

 block1_conv2 (Conv2D)       (None, 32, 32, 64)        36928

 block1_pool (MaxPooling2D)  (None, 16, 16, 64)        0

 block2_conv1 (Conv2D)       (None, 16, 16, 128)       73856

 block2_conv2 (Conv2D)       (None, 16, 16, 128)       147584

 block2_pool (MaxPooling2D)  (None, 8, 8, 128)         0

 block3_conv1 (Conv2D)       (None, 8, 8, 256)         295168

 block3_conv2 (Conv2D)       (None, 8, 8, 256)         590080

 block3_conv3 (Conv2D)       (None, 8, 8, 256)         590080

 block3_pool (MaxPooling2D)  (None, 4, 4, 256)         0

 block4_conv1 (Conv2D)       (None, 4, 4, 512)         1180160

 block4_conv2 (Conv2D)       (None, 4, 4, 512)         2359808

 block4_conv3 (Conv2D)       (None, 4, 4, 512)         2359808

 block4_pool (MaxPooling2D)  (None, 2, 2, 512)         0

 block5_conv1 (Conv2D)       (None, 2, 2, 512)         2359808

 block5_conv2 (Conv2D)       (None, 2, 2, 512)         2359808

 block5_conv3 (Conv2D)       (None, 2, 2, 512)         2359808   

 block5_pool (MaxPooling2D)  (None, 1, 1, 512)         0

=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________

지금 내려받은 신경망 구조에는 분류기 부분(3개의 전결합층)이 빠져있다(include_top 인수의 값을 False로 했기 때문이다). 
출력된 내용에 따르면 이 모델에는 약 1400만개의 학습 가능한 파라미터가 있다. 이들의 가중치는 변경하지 않고 그대로 사용하며 새로 구현한 분류기 부분만 추가한다.
'''
vgg16.trainable=False  # 가중치를 고정해서 앞으로 있을 추가 학습에서 변경되지 않도록 한다.

model = Sequential()
model.add(vgg16)

# 분류기 부분을 새로 구현해서 추가한다. 
model.add(Flatten())  
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))  #  분류 대상 클래스가 10개이므로 유닛이 10개인 소프트맥스층을 추가한다.

model.summary()
'''
_________________________________________________________________
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1, 1, 512)         14714688

 flatten (Flatten)           (None, 512)               0

 dense (Dense)               (None, 100)               51300

 dense_1 (Dense)             (None, 10)                1010

=================================================================
Total params: 14,766,998
Trainable params: 52,310
Non-trainable params: 14,714,688
_________________________________________________________________
'''
print(len(model.weights))  # 30     
print(len(model.trainable_weights))  # 4