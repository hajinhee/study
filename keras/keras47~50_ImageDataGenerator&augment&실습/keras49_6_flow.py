from tensorflow.keras import datasets
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
# import warnings

# warnings.filterwarnings(action='ignore')    #warining 메시지를 무시하고 필터로 걸러줌

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(    
    rescale=1./255,                    
    horizontal_flip=True,               
    #vertical_flip=True,                                      
    width_shift_range=0.1,            
    height_shift_range=0.1,   
    #rotation_range=5,               
    zoom_range=0.1,                 
    #shear_range=0.7,                    
    fill_mode='nearest'        
)

test_datagen = ImageDataGenerator(rescale=1./255)

augment_size = 40000
randidx = np.random.randint(x_train.shape[0],size=augment_size)

x_augmented = x_train[randidx].copy()     
y_augmented = y_train[randidx].copy()      

x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)

xy_train = train_datagen.flow(
    x_augmented, y_augmented,
    batch_size=32,    
    shuffle=False
)

x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
xy_test = test_datagen.flow(
    x_test,y_test,batch_size=32
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(28,28,1)))
model.add(Conv2D(64, (2,2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])     # sparse_cate~~ 해주면 원핫인코딩 안해줘도 돌아간다.
model.fit_generator(xy_train,epochs=10,steps_per_epoch=len(xy_train)//50)  

#4. 훈련,평가
loss = model.evaluate_generator(xy_test)
print(loss)

# sparse_cate~~쓰는것만으로 categorical하게 y값을 연산해서 output값까지 categorical하게 주고 그 이후 argmax쓰면 output다시 int로 바꿔준다.
ypred = model.predict_generator(x_test)
ypred_int = np.argmax(ypred,axis=1)
print(ypred[:10])
print(ypred_int[:10])