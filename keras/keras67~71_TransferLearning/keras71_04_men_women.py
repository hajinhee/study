from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception,VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings, os
from icecream import ic
# warnings.filterwarnings(action='ignore')
from PIL import Image

#1. load images 
path = 'keras/data/images/men_or_women/'
men = os.listdir(path+'/men')  # 1418
women = os.listdir(path+'/women')  # 1912                
ic(len(men), len(women))   

#################################### Save ####################################
# basic_datagen = ImageDataGenerator(rescale=1/255., validation_split=0.2)

# m = []
# w = []
# for i in men:
#     m.append(np.array(Image.open(f'{path}/men/{i}').convert('RGB').resize((100,100)))) 
# for i in women:
#     w.append(np.array(Image.open(f'{path}/women/{i}').convert('RGB').resize((100,100))))  

# # list -> np.array    
# mm = np.array(m)
# ww = np.array(w)

# mw_augment_datagen = ImageDataGenerator(      
#     horizontal_flip=True, 
#     rotation_range=3,       
#     width_shift_range=0.3, 
#     height_shift_range=0.3, 
#     zoom_range=(0.3),       
#     fill_mode='nearest',  
# )

# m_augmented_size = 2000-len(mm)  # 데이터 2000개 맞추기 위해 
# w_augmented_size = 2000-len(ww)     
# m_randidx = np.random.randint(len(mm), size=m_augmented_size)  # 개 중에서 랜덤으로 개의 인덱스 m_randidx에 할당
# w_randidx = np.random.randint(len(ww), size=w_augmented_size)  

# m_augmented = mw_augment_datagen.flow(
#     mm[m_randidx], np.ones(m_augmented_size),  # np.ones(shape, dtype, order): ones는 zeros와 마찬가지로 1로 가득찬 array를 생성 
#     batch_size=100, 
#     shuffle=False,
#     save_to_dir=f'{path}/men/',
#     save_prefix='m_aug_',
#     save_format='jpg'
# ).next()[0]  # x_data(men)

# w_augmented = mw_augment_datagen.flow(
#     ww[w_randidx], np.zeros(w_augmented_size),  
#     batch_size=100, 
#     shuffle=False,
#     save_to_dir=f'{path}/women/',
#     save_prefix='w_aug',
#     save_format='jpg'
# ).next()[0]  # x_data(women) -----------> men, women x_data 증폭 완료 후 'path'에 저장

# mw_train = basic_datagen.flow_from_directory(      
#     path,  # 기존 데이터와 증폭된 데이터 모두 저장된 상태
#     target_size=(100, 100),
#     batch_size=100, 
#     class_mode='binary',  
#     classes=['men','women'],
#     subset='training'        
# )   # Found 2799 images belonging to 2 classes.

# mw_test = basic_datagen.flow_from_directory(
#     path,
#     target_size=(100, 100),
#     batch_size=100,
#     class_mode='binary',  
#     classes= ['men','women'],
#     subset='validation'
# )   # Found 698 images belonging to 2 classes.

# np.save(path + '/mw_train_x', arr=mw_train[0][0])  # x_train 
# np.save(path + '/mw_train_y', arr=mw_train[0][1])  # y_train
# np.save(path + '/mw_test_x', arr=mw_test[0][0])  # x_test
# np.save(path + '/mw_test_y', arr=mw_test[0][1])  # y_test

#################################### Load ####################################
#1. load data
x_train = np.load(path + '/mw_train_x.npy')  # (100, 100, 100, 3)
y_train = np.load(path + '/mw_train_y.npy')  # (100,)
x_test = np.load(path + '/mw_test_x.npy')  # (100, 100, 100, 3)
y_test = np.load(path + '/mw_test_y.npy')  # (100,)

ic(x_train.shape, y_train.shape, x_test.shape, y_test.shape) 

#2. modeling(VGG19, Xception)
model_list = [VGG19(weights='imagenet', include_top=False, input_shape=(100,100,3), pooling='max', classifier_activation='sigmoid'),
              Xception(weights='imagenet', include_top=False, input_shape=(100,100,3), pooling='max', classifier_activation='sigmoid')]

for model in model_list:
    TL_model = model
    model = Sequential()
    model.add(TL_model) 
    # classifier    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #3. compile, train
    optimizer = Adam(learning_rate=0.0001)  # 1e-4     
    lr=ReduceLROnPlateau(monitor='val_acc', patience=3, mode='max', factor=0.1, min_lr=0.00001, verbose=False)
    es = EarlyStopping(monitor='val_acc', patience=5, mode='max', verbose=1, restore_best_weights=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='acc')
    
    start = time.time()
    model.fit(x_train, y_train, batch_size=5, epochs=10, validation_split=0.2, callbacks=[lr, es], verbose=1)
    end = time.time()
    
    #4. evaluate
    loss, acc = model.evaluate(x_test, y_test, batch_size=10, verbose=True)
    print(f"Model : {TL_model.name}")
    print(f"Time : {round(end-start, 4)}")
    print(f"loss : {round(loss, 4)}")
    print(f"Acc : {round(acc, 4)}")

'''
Model : vgg19
Time : 10.5192
loss : 0.6893
Acc : 0.54

Model : xception
Time : 12.3126
loss : 0.6952
Acc : 0.53
'''