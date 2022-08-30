from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19, Xception, ResNet50, ResNet101, InceptionV3
from tensorflow.keras.applications import InceptionResNetV2, DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.datasets import cifar100
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings
warnings.filterwarnings(action='ignore')
from icecream import ic

#1. load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
ic(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

'''
normalize 대신 preprocess_input 사용
x_train = x_train/255.  -->   x_train = preprocess_input(x_train) 
x_test = x_test/255.    -->   x_test = preprocess_input(x_test)  
'''

# load pretrained model 
model_list = [VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3)),
              # Xception(weights='imagenet', include_top=False, input_shape=(32,32,3)),    # minsize = 75 
              ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3)),
              ResNet101(weights='imagenet', include_top=False, input_shape=(32,32,3)),
              # InceptionV3(weights=None, include_top=False, input_shape=(32,32,3)),       # minsize = 75
              # InceptionResNetV2(weights=None, include_top=False, input_shape=(32,32,3)), # minsize = 75
              DenseNet121(weights='imagenet', include_top=False, input_shape=(32,32,3)),
              MobileNetV2(weights='imagenet', include_top=False, input_shape=(32,32,3)),
              NASNetMobile(weights=None, include_top=False, input_shape=(32,32,3)),        # weights=None하면 input_shape=(32,32,3) 사용 가능
              EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32,32,3))]

#2. modeling
for model in model_list:
    # preprocess
    x_train = preprocess_input(x_train)   
    x_test = preprocess_input(x_test)   
    TL_model = model
    TL_model.trainable = True
    # classifier 
    model = Sequential()
    model.add(TL_model)    
    model.add(GlobalAveragePooling2D())  
    model.add(Dense(1024))
    model.add(Dense(512))
    model.add(Dense(100, activation='softmax'))

    #3. compile, train
    optimizer = Adam(learning_rate=0.0001)  # 1e-4     
    lr=ReduceLROnPlateau(monitor='val_acc', patience = 3, mode='max', factor = 0.1, min_lr=0.00001, verbose=False)
    es = EarlyStopping(monitor='val_acc', patience=15, mode='max', verbose=1, restore_best_weights=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics='acc')
    
    start = time.time()
    model.fit(x_train, y_train, batch_size=200, epochs=100, validation_split=0.2, callbacks=[lr, es], verbose=True)
    end = time.time()
    
    #4. evaluate
    loss, acc = model.evaluate(x_test, y_test, batch_size=100, verbose=True)
    print(f"Time : {round(end-start, 4)}")
    print(f"loss : {round(loss, 4)}")
    print(f"Acc : {round(acc, 4)}")
    
'''
모델명 : vgg19
Time : 803.8299
loss : 4.9
Acc : 0.5582

모델명 : resnet50
Time : 1093.0532
loss : 3.8416
Acc : 0.4741

모델명 : resnet101
Time : 1752.1339
loss : 3.954
Acc : 0.4724

모델명 : densenet121
Time : 1220.9396
loss : 3.1168
Acc : 0.5388

모델명 : mobilenetv2_1.00_224
Time : 653.221
loss : 4.5997
Acc : 0.4188

모델명 : NASNet
Time : 496.2634
loss : 44.2563
Acc : 0.0114

모델명 : efficientnetb0
Time : 954.3006
loss : 2.9445
Acc : 0.4872
'''