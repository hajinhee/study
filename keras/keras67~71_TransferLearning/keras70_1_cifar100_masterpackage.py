from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import VGG19, Xception, ResNet50, ResNet101, InceptionV3
from tensorflow.keras.applications import InceptionResNetV2, DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0
from tensorflow.keras.datasets import cifar100
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings
warnings.filterwarnings(action='ignore')
from icecream import ic

#1. load data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
ic(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)
ic(len(np.unique(y_test)))  # 100 

# normalize
x_train = x_train/255.
x_test = x_test/255.

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
    print(f"모델명 : {model.name}")
    TL_model = model
    TL_model.trainable = True
    model = Sequential()
    model.add(TL_model)    
    # classifier 
    model.add(GlobalAveragePooling2D())  
    model.add(Dense(1024))
    model.add(Dense(512))
    model.add(Dense(100, activation='softmax'))
    
    #3. compile, train
    optimizer = Adam(learning_rate=0.0001)  # 1e-4     
    lr=ReduceLROnPlateau(monitor='val_acc', patience=3, mode='max', factor=0.1, min_lr=0.00001, verbose=False)
    es = EarlyStopping(monitor='val_acc', patience=15, mode='max', verbose=1, restore_best_weights=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics='acc')
    
    start = time.time()
    model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2, callbacks=[lr, es], verbose=True)
    end = time.time()
    
    #4. evaluate
    loss, acc = model.evaluate(x_test, y_test, batch_size=100, verbose=True)
    print(f"Time : {round(end-start, 4)}")
    print(f"loss : {round(loss, 4)}")
    print(f"Acc : {round(acc, 4)}")

    
'''
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