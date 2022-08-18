from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

#1.데이터 로드 및 전처리

# 데이터를 train test없이 그냥 말과 사람 뭉탱이로 제공해주었다.
# -> 말과 뭉탱이를 각각 말 train test 사람 train test로 일정 비율로 나누고 싶다.
# -> 왜 why train_set은 이미지제네레이터에서 변환후 불러오고 test_set은 그냥 불러와서 fit과 evaulate 하고싶기때문이다.

path = "../_data/image/rps"      

rock = os.listdir(path+'/1_rock')             
paper = os.listdir(path+'./2_paper')          
scissors = os.listdir(path+'./3_scissors')    
#print(len(rock),len(paper),len(scissors))  # 각각 840장!

rock_train, rock_test = train_test_split(rock, train_size=0.8, shuffle=True, random_state=49)
paper_train, paper_test = train_test_split(paper, train_size=0.8, shuffle=True, random_state=49)
scissors_train, scissors_test = train_test_split(scissors, train_size=0.8, shuffle=True, random_state=49)
#print(len(rock_train),len(rock_test)) 672 168 개 확인.

os.makedirs(f'{path}/training_set/1_rock', exist_ok=True)
os.makedirs(f'{path}/training_set/2_paper', exist_ok=True)
os.makedirs(f'{path}/training_set/3_scissors', exist_ok=True)
os.makedirs(f'{path}/test_set/1_rock', exist_ok=True)
os.makedirs(f'{path}/test_set/2_paper', exist_ok=True)
os.makedirs(f'{path}/test_set/3_scissors', exist_ok=True)

for i in rock_train:
    shutil.copy2(f'{path}/1_rock/'+i,f'{path}/training_set/1_rock/'+i)
for i in paper_train:
    shutil.copy2(f'{path}/2_paper/'+i,f'{path}/training_set/2_paper/'+i)
for i in scissors_train:
    shutil.copy2(f'{path}/3_scissors/'+i,f'{path}/training_set/3_scissors/'+i)
for i in rock_test:
    shutil.copy2(f'{path}/1_rock/'+i,f'{path}/test_set/1_rock/'+i)
for i in paper_test:
    shutil.copy2(f'{path}/2_paper/'+i,f'{path}/test_set/2_paper/'+i)
for i in scissors_test:
    shutil.copy2(f'{path}/3_scissors/'+i,f'{path}/test_set/3_scissors/'+i)    


'''
train_datagen = ImageDataGenerator(    
    rescale=1./255,                    
    horizontal_flip=True,               
    vertical_flip=True,                                      
    width_shift_range=0.1,            
    height_shift_range=0.1,   
    rotation_range=5,               
    zoom_range=1.2,                 
    shear_range=0.7,                    
    fill_mode='nearest',
    validation_split=0.2          
)

test_datagen = ImageDataGenerator(
    rescale=1./255                      
)

b = 5

xy_train_train = train_datagen.flow_from_directory(      
    '../_data/image/horse-or-human/training_set/',
    target_size = (200, 200),                                                                       
    batch_size=b,                                   
    class_mode='categorical',        
    shuffle=True,  seed=66,  
    subset='training'
)   

xy_train_val = train_datagen.flow_from_directory(      
    '../_data/image/horse-or-human/training_set/',
    target_size = (200, 200),                                                                       
    batch_size=b,                                   
    class_mode='categorical',        
    shuffle=True,   seed=66,   
    subset='validation'
) 

xy_test = test_datagen.flow_from_directory(         
    '../_data/image/horse-or-human/test_set/',
    target_size=(200,200),
    batch_size=b,
    class_mode='categorical',                            
)  

if len(xy_train_train)%b == 0:
    spe = len(xy_train_train)//b
else:
    spe = len(xy_train_train)//b + 1

if len(xy_train_val)%b == 0:
    vs = len(xy_train_val)//b
else:
    vs = len(xy_train_val)//b + 1

# xy_train_train,xy_train_val,xy_test    3개의 파일이 생긴다

#print(len(xy_train_train),len(xy_train_val),len(xy_test))   # batch  5에 각각 132 33 42  207 x  5 = 1035    나머지연산 감안하며 딱 맞다.
                                                            # batch 10에 각각  66 17 21  104 x 10 = 1040    드디어 비로소 제대로 된 모델링 시작가능.


#2. 모델링
model = Sequential()
model.add(Conv2D(32, (2,2), padding='same',input_shape=(200,200,3), activation='relu'))
model.add(MaxPool2D(2))                                                     # 50,50,32
model.add(Conv2D(16, (4,4), activation='relu'))                            # 72,72,16
model.add(Flatten())
model.add(Dense(36, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(2,activation='softmax'))

#3. 컴파일,훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor = "acc", patience=50, mode='max',verbose=1,restore_best_weights=True)
model.fit(xy_train_train,epochs=10000,steps_per_epoch=spe,validation_data=xy_train_val,validation_steps=vs,callbacks=[es])            
                     
#4. 평가,예측.

loss = model.evaluate(xy_test)  
print(loss)                     # [0.1567889153957367, 0.946601927280426]

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
    
'''
