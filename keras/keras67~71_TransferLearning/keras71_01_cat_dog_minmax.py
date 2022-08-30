from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception,VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from icecream import ic
import numpy as np, time, warnings, os
# warnings.filterwarnings(action='ignore')

#1. load images 
path = 'keras/data/images/cat_or_dog'
train_cats = os.listdir(path+'/train/dogs')  # 201      
train_dogs = os.listdir(path+'/train/cats')  # 201   
test_cats = os.listdir(path+'/test/cats')  # 101
test_dogs = os.listdir(path+'/test/dogs')  # 101

ic(len(train_cats), len(train_dogs), len(test_cats), len(test_dogs)) 


#################################### Save ####################################
# img_datagen = ImageDataGenerator(rescale= 1/255.)

# train_datagen = img_datagen.flow_from_directory(      
#     path + '/train/',
#     target_size = (100, 100),                                                                       
#     batch_size=100,                                   
#     class_mode='binary',  
#     classes= ['cats', 'dogs']  # cat=0, dog=1
# )   # Found 402 images belonging to 2 classes.

# test_datagen = img_datagen.flow_from_directory(         
#     path + '/test/',
#     target_size=(100, 100),
#     batch_size=100,
#     class_mode='binary',  
#     classes= ['cats', 'dogs']  # cat=0, dog=1
# )   # Found 202 images belonging to 2 classes.

# np.save(path + '/train/catdog_train_x', arr=train_datagen[0][0])  # x_train
# np.save(path + '/train/catdog_train_y', arr=train_datagen[0][1])  # y_train  
# np.save(path + '/test/catdog_test_x', arr=test_datagen[0][0])  # x_test
# np.save(path + '/test/catdog_test_y', arr=test_datagen[0][1])  # y_test

#################################### Load ####################################
#1. load data
x_train = np.load(path + '/train/catdog_train_x.npy')      
y_train = np.load(path + '/train/catdog_train_y.npy')     
x_test = np.load(path + '/test/catdog_test_x.npy')       
y_test = np.load(path + '/test/catdog_test_y.npy')

ic(x_train.shape, y_train.shape, x_test.shape, y_test.shape) 

#2. modeling(VGG19, Xception)
model_list = [VGG19(weights='imagenet', include_top=False, input_shape=(100,100,3), pooling='max', classifier_activation='sigmoid'),
              Xception(weights='imagenet', include_top=False, input_shape=(100,100,3), pooling='avg', classifier_activation='sigmoid')]
'''
여기서 파라미터 pooling='max or avg'에 주목한다. 이 파라미터는 합성곱층의 출력에 전역 평균 풀링을 적용하라는 의미다. 
따라서 모델의 출력도 2차원 텐서 형태가 된다.
여기서는 전결합층 입력을 위해 사용하는 1차원 변환층(Flatten)을 대체하는 역할을 한다. 
pooling='max' ->  global_max_pooling2d
pooling='avg' ->  global_average_pooling2d
'''

for model in model_list:
    TL_model = model
    TL_model.summary()
    model = Sequential()
    model.add(TL_model) 
    # classifier 
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64,  activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #3. compile, train
    optimizer = Adam(learning_rate=0.0001)  # 1e-4     
    lr=ReduceLROnPlateau(monitor='val_acc', patience = 3, mode='max', factor = 0.1, min_lr=0.00001, verbose=False)
    es = EarlyStopping(monitor='val_acc', patience=15, mode='max', verbose=1, restore_best_weights=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='acc')
    
    start = time.time()
    model.fit(x_train, y_train, batch_size=50, epochs=10, validation_split=0.2, callbacks=[lr, es], verbose=1)
    end = time.time()
    
    #4. evaluate
    loss, acc = model.evaluate(x_test, y_test, batch_size=100, verbose=True)
    print(f"Time : {round(end-start, 4)}")
    print(f"loss : {round(loss, 4)}")
    print(f"Acc : {round(acc, 4)}")
    
'''
모델명 : vgg19
Time : 13.7706
loss : 0.5743
Acc : 0.69

모델명 : xception
Time : 11.4083
loss : 0.6947
Acc : 0.58
'''
