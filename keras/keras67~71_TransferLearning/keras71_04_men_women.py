from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception,VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings, os
# warnings.filterwarnings(action='ignore')
from PIL import Image

path = 'D:\_data\image_classification\men_women'

# save 구간 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 나중에 사진 변환하고 저장할때 비교를 쉽게하기 위해 기존 사진들의 이름을 따서 저장.
# men = os.listdir(path+'/men')         
# women = os.listdir(path+'/women')                  

# print(len(men),len(women))    # 1418 1912
# 증폭시켜서 장수도 맞춰주고, 원래는 디텍션해서 사람같지 않은 사진은 삭제해야하는데 이건 skip하겠다.

# basic_datagen = ImageDataGenerator(rescale=1/255.,validation_split=0.2)
'''
# 증폭해서 각각 2000장으로 늘려서 4000장 가자.
#일단 이미지를 numpy형태로 변환 후, 이미지제너레이트해야 증폭까지 가능하다.

m = []
for i in men:
    m.append(np.array(Image.open(f'{path}/men/{i}').convert('RGB').resize((100,100))))    #반복문 써서 1418장 변환
# Image.open으로 이미지를 불러오고 컬러형태이기때문에 convert로 RGB계산해서 불러오고 300,300으로 사이즈 조정해준다.
mm = np.array(m)
w = []
for i in women:
    w.append(np.array(Image.open(f'{path}/women/{i}').convert('RGB').resize((100,100))))  #반복문 써서 1912장 변환
ww = np.array(w)

mw_augment_datagen = ImageDataGenerator(        # 남녀 사진 변환용      
    horizontal_flip=True,  
    rotation_range=3,       
    width_shift_range=0.3, 
    height_shift_range=0.3, 
    zoom_range=(0.3),       
    fill_mode='nearest',  
    #samplewise_center=gen_keras_mean_norm_sample_wise
)

m_augmented_size = 2000 - len(mm)        # 582개
w_augmented_size = 2000 - len(ww)        # 88개
m_randidx = np.random.randint(len(mm),size=m_augmented_size)    # 1~1418개중에서 임의로 582개의 숫자 뽑아서 저장
w_randidx = np.random.randint(len(ww),size=w_augmented_size)    # 1~1912개중에서 임의로  88개의 숫자 뽑아서 저장

m_augmented = mw_augment_datagen.flow(
    mm[m_randidx], np.ones(m_augmented_size),
    batch_size=m_augmented_size, shuffle=False,
    save_to_dir=f'{path}/men/',save_prefix='m_aug_',
    save_format='jpg'
).next()[0] # 변환 후 다시 x값만 저장.
w_augmented = mw_augment_datagen.flow(
    ww[w_randidx], np.zeros(w_augmented_size),
    batch_size=w_augmented_size, shuffle=False,
    save_to_dir=f'{path}/women/',save_prefix='w_aug',
    save_format='jpg'
).next()[0]
'''
# 원래 증폭할때 이름까지 값 따올려했는데 지금은 시간관계상 pass하겠다. 일단은 증폭완료.
'''
mw_train = basic_datagen.flow_from_directory(      
    path,
    target_size = (100, 100),                                                                       
    batch_size=100000,                                   
    class_mode='binary',  
    classes= ['men','women'],
    subset = 'training'        
)   

mw_test = basic_datagen.flow_from_directory(         
    path,
    target_size=(100,100),
    batch_size=1000000,
    class_mode='binary',  
    classes= ['men','women'],
    subset='validation'                              
) 

np.save(path + '/mw_train_x', arr=mw_train[0][0])    
np.save(path + '/mw_train_y', arr=mw_train[0][1])    
np.save(path + '/mw_test_x', arr=mw_test[0][0])      
np.save(path + '/mw_test_y', arr=mw_test[0][1]) 
'''
# load 구간 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

x_train = np.load(path + '/mw_train_x.npy')      
y_train = np.load(path + '/mw_train_y.npy')     
x_test = np.load(path + '/mw_test_x.npy')       
y_test = np.load(path + '/mw_test_y.npy')
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)  # (3184, 100, 100, 3) (3184,) (795, 100, 100, 3) (795,)

model_list = [VGG19(weights='imagenet', include_top=False, input_shape=(100,100,3),pooling='max',classifier_activation='sigmoid'),
              Xception(weights='imagenet', include_top=False, input_shape=(100,100,3),pooling='max',classifier_activation='sigmoid')]

for model in model_list:
    print(f"모델명 : {model.name}")
    TL_model = model
    TL_model.trainable = True
    model = Sequential()
    model.add(TL_model)    
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    optimizer = Adam(learning_rate=0.0001)  # 1e-4     
    lr=ReduceLROnPlateau(monitor= "val_acc", patience = 3, mode='max',factor = 0.1, min_lr=0.00001,verbose=False)
    es = EarlyStopping(monitor ="val_acc", patience=5, mode='max',verbose=1,restore_best_weights=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='acc')
    
    start = time.time()
    model.fit(x_train,y_train,batch_size=5,epochs=1000,validation_split=0.2,callbacks=[lr,es], verbose=1)
    end = time.time()
    
    loss, Acc = model.evaluate(x_test,y_test,batch_size=10,verbose=False)
    
    print(f"Time : {round(end - start,4)}")
    print(f"loss : {round(loss,4)}")
    print(f"Acc : {round(Acc,4)}")
'''
모델명 : vgg19
Time : 511.6331
loss : 0.5118
Acc : 0.8642

모델명 : xception
Time : 400.7548
loss : 0.7422
Acc : 0.8138
'''