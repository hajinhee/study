import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

# model = VGG16()
model = VGG16(weights=None, include_top=True, classes=100, pooling='max') 
# model = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3), classes=100, pooling='max') 
'''
weight('imagenet'): 로딩할 가중치. 처음부터 훈련시키는데 관심이 있다면 None을 통해 사전에 훈련된 가중치를 사용하지 않아도 된다.
include_top(True): 분류기 부분(전결합층)의 가중치를 내려받을지 여부로 개별 문제에 적합하게 되어있다면 포함한다. False인 경우 input_shape=img_size 지정이 필요하다.
input_shape(None): 입력 레이어를 변경할 경우 모델이 가져올 것으로 기대하는 이미지의 크기
classes(1000): 출력 벡터와 같은 해당 모델의 클래스의 수  
pooling(None): 출력 레이어의 새로운 세트를 훈련시킬 때 사용하는 풀링 타입
input_tensor(None): 서로 다른 크기의 새로운 데이터에 모델을 맞추기 위한 새로운 입력 레이어

'''
model.summary()
########################## include_top = True ################################
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



