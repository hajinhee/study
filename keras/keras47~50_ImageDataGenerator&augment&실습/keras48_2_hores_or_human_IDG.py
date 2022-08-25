from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import shutil

#1. 데이터 로드 및 전처리
train_datagen = ImageDataGenerator(    
    rescale=1./255,  # 스케일링                   
    horizontal_flip=True,  # 상하반전
    vertical_flip=True,  # 좌우반전
    width_shift_range=0.1,  # 좌우이동
    height_shift_range=0.1,  # 상하이동
    rotation_range=5,  # 회전
    zoom_range=1.2,  # 확대
    shear_range=0.7,  # 기울기
    fill_mode='nearest',
    validation_split=0.2  # 검증데이터 분할
)

test_datagen = ImageDataGenerator(
    rescale=1./255                      
)

xy_train = train_datagen.flow_from_directory(      
    'keras/data/images/horse_or_human/train/',
    target_size = (200, 200),                                                                       
    batch_size=3,                                   
    class_mode='categorical',        
    shuffle=True,  
    seed=42,  
    subset='training'  # set as training data
)   # Found 822 images belonging to 2 classes.

xy_val = train_datagen.flow_from_directory(      
    'keras/data/images/horse_or_human/train/',
    target_size = (200, 200),                                                                       
    batch_size=3,                                   
    class_mode='categorical',        
    shuffle=True,   
    seed=42,   
    subset='validation'  # set as validation data
)   # Found 205 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(         
    'keras/data/images/horse_or_human/test/',
    target_size=(200, 200),
    batch_size=3,
    class_mode='categorical',                            
)   # Found 256 images belonging to 2 classes.

print(len(xy_train), len(xy_val), len(xy_test))   # 274, 69, 86
print(xy_train.image_shape, xy_val.image_shape, xy_test.image_shape)  # (200, 200, 3) (200, 200, 3) (200, 200, 3)
                                                  

#2. 모델링
model = Sequential()
model.add(Conv2D(32, (2, 2), padding='same', input_shape=(200, 200, 3), activation='relu'))
model.add(MaxPool2D(2))                                                    
model.add(Conv2D(16, (4, 4), activation='relu'))                
model.add(Flatten())
model.add(Dense(36, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='acc', patience=50, mode='max', verbose=1, restore_best_weights=True)
model.fit(xy_train, epochs=1, steps_per_epoch=len(xy_train), validation_data=xy_val, validation_steps=len(xy_val), callbacks=[es])            
                     
#4. 평가, 예측
loss = model.evaluate(xy_test, batch_size=1)  
print(' [loss]: ', loss[0], '\n','[acc]: ', loss[1])     

'''
 [loss]:  0.0 
 [acc]:  0.5
'''

img_path = 'keras/data/images/sample.jpg'

# 이미지 확인
# image_ = plt.imread(str(img_path))
# plt.title("Test Case")
# plt.imshow(image_)
# plt.axis('Off')
# plt.show()

def load_my_image(img_path):
    img = image.load_img(str(img_path), target_size=(200, 200))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /=255.
    return img_tensor
 
img = load_my_image(img_path)
img_pred = model.predict(img)
print(img_pred)

# {'hores': 0, 'human': 1}
if(img_pred[0][0]<=0.5):
    hores = 100 - img_pred[0][0]*100
    print(f"당신은 {round(hores,2)} % 확률로 horse 입니다")
elif(img_pred[0][0]>0.5):
    human = img_pred[0][0]*100
    print(f"당신은 {round(human,2)} % 확률로 human 입니다")
else:
    print("ERROR")