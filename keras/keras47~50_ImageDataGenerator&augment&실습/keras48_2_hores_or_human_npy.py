import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.backend import binary_crossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from icecream import ic
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

#1. 데이터
x_train = np.load('keras/save/npy/keras48_2_train_x.npy')  # (3, 200, 200, 3)
y_train = np.load('keras/save/npy/keras48_2_train_y.npy')  # (3,)
x_test = np.load('keras/save/npy/keras48_2_test_x.npy')  # (3, 200, 200, 3)
y_test = np.load('keras/save/npy/keras48_2_test_y.npy')  # (3,)

#2. 모델
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=(200, 200, 3)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu')) 
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', patience=10, mode='max', verbose=1) 
hist = model.fit(x_train, y_train, epochs=20, batch_size=32,
                 validation_split=0.2, callbacks=[es])  

# 4. 평가
loss = model.evaluate(x_test, y_test)                 
print('[loss]:', loss[0], '\t', '[acc]:', loss[1])

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:',val_acc [-1])

'''
[loss]: 22.046951293945312       [acc]: 0.3333333432674408
loss: 0.06981496512889862
val_loss: 0.6588208675384521
acc: 1.0
val_acc: 1.0
'''

#############prtdict##############
pic_path = 'keras/data/images/sample.jpg'
model_path = './_save/keras48_2_save.h5'

def load_my_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(200, 200))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /=255.
    
    if show:
        plt.imshow(img_tensor[0])    
        plt.append('off')
        plt.show()
    return img_tensor

if __name__ == '__main__':
    model = load_model(model_path)
    new_img = load_my_image(pic_path)
    img_pred = model.predict(new_img)
    # {'hores': 0, 'human': 1}
    if(img_pred[0][0]<=0.5):
        print(f"당신은 {round(img_pred[0][0]*100, 2)} % 확률로 horse 입니다")
    elif(img_pred[0][0]>0.5):
        print(f"당신은 {round(img_pred[0][0]*100, 2)} % 확률로 human 입니다")
    else:
        print("ERROR")