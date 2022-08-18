from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
#1.데이터 로드 및 전처리

# train_datagen = ImageDataGenerator(    
#     rescale=1./255,                    
#     horizontal_flip=True,               
#     vertical_flip=True,                                      
#     width_shift_range=0.1,            
#     height_shift_range=0.1,   
#     rotation_range=5,               
#     zoom_range=1.2,                 
#     shear_range=0.7,                    
#     fill_mode='nearest'         
# )

# test_datagen = ImageDataGenerator(
#     rescale=1./255                      
# )

# xy_train = train_datagen.flow_from_directory(      
#     '../_data/image/horse-or-human/training_set/',
#     target_size = (200, 200),                                                                       
#     batch_size=10000000000,                                   
#     class_mode='categorical',        
#     shuffle=True,    
# )   

# xy_test = test_datagen.flow_from_directory(         
#     '../_data/image/horse-or-human/test_set/',
#     target_size=(200,200),
#     batch_size=10000000000,
#     class_mode='categorical',                            
# )  

# 배치사이즈를 이빠이줘서 1개씩 그냥 통짜로 그 안에 사진들을 다 저장하고 넘파이형태로 파일저장해주는데,
# 

#******* 매우중요.  
'''
이미지제너레이터란 결국 2가지 이상의 이미지 데이터 세트를 (사람or말,강아지or고양이,가위or바위or보)를 폴더째로 불러와서 엮어서 
각 1장마다 그것을 수치화한데이터 + 그것에 대한 라벨값을 tuple형태로 저장해준다.
사진의 종류가 워낙 방대하니까 batch사이즈로 5장씩 묶어서 165개의 묶음으로 만들어주고 읽어봐 그냥 아래 내용 읽어봐 읽고 생각 이해.

print(type(xy_train))
print(len(xy_train))
print(type(xy_train[0]))
print(type(xy_train[0][0]))
print(xy_train[0][0].shape)
print(type(xy_train[0][1]))
print(xy_train[0][1].shape)

<class 'keras.preprocessing.image.DirectoryIterator'>
165
<class 'tuple'>
<class 'numpy.ndarray'>
(5, 200, 200, 3)
<class 'numpy.ndarray'>
(5, 2)

'''
# np.save('../_data/_save_npy/keras48_2_train_x.npy', arr=xy_train[0][0])    
# np.save('../_data/_save_npy/keras48_2_train_y.npy', arr=xy_train[0][1])    
# np.save('../_data/_save_npy/keras48_2_test_x.npy', arr=xy_test[0][0])      
# np.save('../_data/_save_npy/keras48_2_test_y.npy', arr=xy_test[0][1]) 

x_train = np.load('../_data/_save_npy/keras48_2_train_x.npy')      #(821, 200, 200, 3)
y_train = np.load('../_data/_save_npy/keras48_2_train_y.npy')      #(821,2)
x_test = np.load('../_data/_save_npy/keras48_2_test_x.npy')        #(206, 200, 200, 3)
y_test = np.load('../_data/_save_npy/keras48_2_test_y.npy')        #(206,2) 카테고리컬 해줘서 2! 

#print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)    #(821, 200, 200, 3) (821, 2) (206, 200, 200, 3) (206, 2)


#2. 모델링
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same',input_shape=(200,200,3), activation='relu'))
model.add(MaxPool2D(2))                                                     
model.add(Conv2D(16, (4,4), activation='relu'))                           
model.add(Flatten())
model.add(Dense(36, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(2,activation='softmax'))

#3. 컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor = "val_acc", patience=50, mode='max',verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=1,batch_size=5,validation_split=0.2,callbacks=[es])    

#4. 평가,예측

loss = model.evaluate(x_test,y_test, batch_size=5)
print(' loss : ', loss[0])          #    loss :  0.6956891417503357
print(' acc : ', loss[1])           #    acc :  0.534849226474762


pic_path = '../_data/image/증사.jpg'

def load_my_image(img_path,show=False):
    img = image.load_img(img_path, target_size=(200,200))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /=255.

    if show:
            plt.imshow(img_tensor[0])    
            plt.show()
    
    return img_tensor

img = load_my_image(pic_path)

pred = model.predict(img)

horseper = round(pred[0][0]*100,1)
humanper = round(pred[0][1]*100,1)
    
    
if pred[0][0] > pred[0][1]:
    print(f'당신은 {horseper}% 의 확률로 말입니다')
else : 
    print(f'당신은 {humanper}% 의 확률로 사람입니다')
