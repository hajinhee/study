# fashion_mnist에 적용해서 데이터 증폭시켜서 해보기.
from tensorflow.keras import datasets
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

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

augment_size = 40000
randidx = np.random.randint(x_train.shape[0],size=augment_size)

x_augmented = x_train[randidx].copy()      
y_augmented = y_train[randidx].copy()     

x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_augmented = train_datagen.flow(
    x_augmented, np.zeros(augment_size),
    batch_size=augment_size, shuffle=False
).next()[0]                                     

x_train = np.concatenate((x_train, x_augmented))   
y_train = np.concatenate((y_train, y_augmented))

#여기서부터 이어붙임. 후 #1단계 삭제. + y값 원핫인코딩은 위에서 안했기때문에 여기서 해줌. 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pandas import get_dummies
from sklearn.metrics import accuracy_score

#1. y값만 원핫인코딩

y_train = get_dummies(y_train)
y_test = get_dummies(y_test)

#2. 모델링

model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2), input_shape=(28,28,1)))  
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(10,(2,2), activation='relu'))
model.add(Dropout(0.2))
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=100,validation_split=0.2, callbacks=[es])

#4. 평가 예측
loss = model.evaluate(x_test,y_test, batch_size=10)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

y_pred = model.predict(x_test, batch_size=10)

y_test = y_test.values      # 아 이거하나 잡느라고 30분이 걸리네. 앞으로는 꼭 필요한거 아니면 numpy쓰기.

y_test_int = np.argmax(y_test,axis=1) 
y_pred_int = np.argmax(y_pred,axis=1)        

acc = accuracy_score(y_test_int,y_pred_int)

print('acc_scroe : ',acc)

# 현재        증폭시키기전 값.
# loss :      0.31123247742652893       0.3209112584590912      0.32448408007621765
# accuracy :  0.8934000134468079        0.8866000175476074      0.888700008392334

# train 60000 -> 100000만으로 증폭시킨 후 .
# loss :      0.32165610790252686       0.3205461800098419
# accuracy :  0.8970999717712402        0.8945000171661377
# acc_score:                            0.8945 다를게 없다 똑같다. 내 생각이맞았음.