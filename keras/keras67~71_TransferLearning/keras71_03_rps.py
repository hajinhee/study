from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception,VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings, os
from icecream import ic
# warnings.filterwarnings(action='ignore')

#1. load images 
path = 'keras/data/images/rps/'
paper = os.listdir(path+'paper')  # 712
rock = os.listdir(path+'rock')  # 726
scissors = os.listdir(path+'scissors')  # 750
ic(len(paper), len(rock), len(scissors))  

#################################### Save ####################################
# img_datagen = ImageDataGenerator(rescale= 1/255.,validation_split=0.2)

# rps_train = img_datagen.flow_from_directory(      
#     path,
#     target_size = (200, 200),                                                                       
#     batch_size=100,                                   
#     class_mode='categorical',  
#     classes= ['paper','rock','scissors'], 
#     subset = 'training'        
# )   # Found 1751 images belonging to 3 classes.

# rps_test = img_datagen.flow_from_directory(         
#     path,
#     target_size=(200, 200),
#     batch_size=100,
#     class_mode='categorical',  
#     classes= ['paper','rock','scissors'],
#     subset='validation'                              
# )   # Found 437 images belonging to 3 classes.

# np.save(path + 'rps_train_x', arr=rps_train[0][0])    
# np.save(path + 'rps_train_y', arr=rps_train[0][1])    
# np.save(path + 'rps_test_x', arr=rps_test[0][0])      
# np.save(path + 'rps_test_y', arr=rps_test[0][1]) 

#################################### Load ####################################
#1. load data
x_train = np.load(path + 'rps_train_x.npy')      
y_train = np.load(path + 'rps_train_y.npy')     
x_test = np.load(path + 'rps_test_x.npy')       
y_test = np.load(path + 'rps_test_y.npy')

ic(x_train.shape,y_train.shape,x_test.shape,y_test.shape) 

#2. modeling(VGG19, Xception)
model_list = [VGG19(weights='imagenet', include_top=False, input_shape=(200,200,3), pooling='max', classifier_activation='softmax'),
              Xception(weights='imagenet', include_top=False, input_shape=(200,200,3), pooling='max', classifier_activation='softmax')]

for model in model_list:
    TL_model = model
    model = Sequential()
    model.add(TL_model) 
    # classifier   
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    #3. compile, train
    optimizer = Adam(learning_rate=0.0001)  # 1e-4     
    lr=ReduceLROnPlateau(monitor='val_acc', patience=3, mode='max', factor=0.1, min_lr=0.00001, verbose=False)
    es = EarlyStopping(monitor='val_acc', patience=10, mode='max', verbose=1, restore_best_weights=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics='acc')
    
    start = time.time()
    model.fit(x_train, y_train, batch_size=10, epochs=10, validation_split=0.2, callbacks=[lr, es], verbose=1)
    end = time.time()
    
    #4. evaluate
    loss, acc = model.evaluate(x_test, y_test, batch_size=10, verbose=True)
    print(f"Model : {TL_model.name}")
    print(f"Time : {round(end-start, 4)}")
    print(f"loss : {round(loss, 4)}")
    print(f"Acc : {round(acc, 4)}")

'''
Model : vgg19
Time : 22.7654
loss : 0.0303
Acc : 0.99

Model : xception
Time : 22.8253
loss : 0.1621
Acc : 0.93
'''