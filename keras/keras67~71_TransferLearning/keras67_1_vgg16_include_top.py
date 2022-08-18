import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

# model = VGG16()
model = VGG16(weights=None, include_top=True, input_shape=(32,32,3), classes=100, pooling='max') #
# model = VGG16(weights="imagenet", include_top=False, input_shape=(32,32,3), classes=100, pooling='max') #

model.summary()
########################## include_top = True ################################
#1. FC layer 원래것 그대로 씀
#2. inputshape=(224,224,3)고정 바꿀 수 없다.

# input_1 (InputLayer)        [(None, 224, 224, 3)]     0
# block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
# ...........................
# fc2 (Dense)                 (None, 4096)              16781312

# predictions (Dense)         (None, 1000)              4097000

# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# _________________________________________________________________


########################## include_top = False ################################
#1. FC layer 삭제 됨    -->     앞으로 커스터마이징을 하겠다.   --> False하고 classes 10,32,100 해줘봐야 의미없다 짤린다.
#2. inputshape를 원하는대로 쓸수있음 + imagenet의 가중치 weights를 가져다 쓸 수 있음.

# input_1 (InputLayer)        [(None, 224, 224, 3)]     0
# block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792
# ...........................
# block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808
# block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________

# print(len(model.weights))               # 레이어 16개 -> len은 32개
# print(len(model.trainable_weights))     # 레이어 16개 -> len은 32개

# 점심과제 : FC layer에 대해 정리   -> Fully Connected layer, Dense처럼 다 연결된 형태의 레이어

