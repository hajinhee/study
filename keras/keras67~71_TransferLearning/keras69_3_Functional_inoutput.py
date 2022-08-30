from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from icecream import ic

#1. load pretrained model 
base_model = InceptionV3(weights='imagenet', include_top=False) 

#2. modeling
x = base_model.output  # base_model 마지막 층의 출력을 다음 층의 입력으로 삼는다.
# classifier 
x = GlobalAveragePooling2D()(x)  # or model.add(Flatten()) 
x = Dense(1024, activation='relu')(x)
output = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:  # 각 층의 가중치를 고정해서 앞으로 있을 추가 학습에서 변경되지 않도록 한다.
    layer.trainable = False
    # or model.trainable = False

model.summary()
ic(base_model.layers)