# VGG19, Xception, ResNet50, ResNet101, InceptionV3 ,InceptionResNetV2, DenseNet121, MobileNetV2, NasNetMobile, EfficeintNetB0,
# fine tuning의 정의 : 전이학습에서 위와 아래 추가로 붙여주는 부분을 개발자가 튜닝하는것.
# pre_trained model : 전이학습처럼 사전 훈련된 모델을 사용한 경우를 일컬음.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19, Xception, ResNet50, ResNet101, InceptionV3
from tensorflow.keras.applications import InceptionResNetV2, DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0
from tensorflow.keras.datasets import cifar100
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings

warnings.filterwarnings(action='ignore')

(x_train,y_train), (x_test,y_test) = cifar100.load_data()

# print(x_train.shape,x_test.shape)               # 32,32,3
# print(len(np.unique(y_test)))                   # 100

x_train = x_train.reshape(50000,32,32,3)/255.
x_test = x_test.reshape(10000,32,32,3)/255.

# aa = NasNetMobile(weights='imagenet')
model_list = [VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3)),
            #   Xception(weights='imagenet', include_top=False, input_shape=(32,32,3)),             minsize = 75 
              ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3)),
              ResNet101(weights='imagenet', include_top=False, input_shape=(32,32,3)),
            #   InceptionV3(weights=None, include_top=False, input_shape=(32,32,3)),          #minsize = 75
            #   InceptionResNetV2(weights=None, include_top=False, input_shape=(32,32,3)),    #minsize = 75
              DenseNet121(weights='imagenet', include_top=False, input_shape=(32,32,3)),
              MobileNetV2(weights='imagenet', include_top=False, input_shape=(32,32,3)),
              NASNetMobile(weights=None, include_top=False, input_shape=(32,32,3)),                 #weights= None하면 32,32,3 사용가능
              EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32,32,3))]

for model in model_list:
    print(f"모델명 : {model.name}")
    TL_model = model
    TL_model.trainable = True
    model = Sequential()
    model.add(TL_model)    
    model.add(GlobalAveragePooling2D())  
    model.add(Dense(1024))
    model.add(Dense(512))
    model.add(Dense(100,activation='softmax'))
    
    optimizer = Adam(learning_rate=0.0001)  # 1e-4     
    lr=ReduceLROnPlateau(monitor= "val_acc", patience = 3, mode='max',factor = 0.1, min_lr=0.00001,verbose=False)
    es = EarlyStopping(monitor ="val_acc", patience=15, mode='max',verbose=1,restore_best_weights=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics='acc')
    
    start = time.time()
    model.fit(x_train,y_train,batch_size=200,epochs=1000,validation_split=0.2,callbacks=[lr,es], verbose=True)
    end = time.time()
    
    loss, Acc = model.evaluate(x_test,y_test,batch_size=100,verbose=False)
    
    print(f"Time : {round(end - start,4)}")
    print(f"loss : {round(loss,4)}")
    print(f"Acc : {round(Acc,4)}")
    
'''
batch 200기준!!! <-----------------------------------------------
모델명 : vgg19
Time : 357.6416
loss : 2.6871
Acc : 0.5823

모델명 : resnet50
Time : 1045.516
loss : 3.9487
Acc : 0.4696

모델명 : resnet101
Time : 1621.1863
loss : 3.8353
Acc : 0.4717

모델명 : densenet121
Time : 1357.5877
loss : 3.1646
Acc : 0.5438

모델명 : mobilenetv2_1.00_224
Time : 921.6555
loss : 4.9463
Acc : 0.416

모델명 : NASNet
Time : 574.9925
loss : 4.6314
Acc : 0.0133

모델명 : efficientnetb0
Time : 669.8647
loss : 3.179
Acc : 0.3278
'''