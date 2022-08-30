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
path = 'keras/data/images/horse_or_human/train/'
horses = os.listdir(path+'horses')  # 500       
humans = os.listdir(path+'humans')  # 527
ic(len(horses), len(humans))   

#################################### Save ####################################
# img_datagen = ImageDataGenerator(rescale= 1/255., validation_split=0.2)

# horsehuman_train = img_datagen.flow_from_directory(      
#     path,
#     target_size=(200, 200),                                                                       
#     batch_size=500,                                   
#     class_mode='binary',  
#     classes= ['horses', 'humans'],  # horses=0, humans=1
#     subset = 'training' 
# )   # Found 822 images belonging to 2 classes.

# horsehuman_test = img_datagen.flow_from_directory(         
#     path,
#     target_size=(200, 200),
#     batch_size=500,
#     class_mode='binary',  
#     classes= ['horses', 'humans'],
#     subset='validation'                              
# )   # Found 205 images belonging to 2 classes.

# np.save(path + 'horsehuman_train_x', arr=horsehuman_train[0][0])  # x_train
# np.save(path + 'horsehuman_train_y', arr=horsehuman_train[0][1])  # y_train
# np.save(path + 'horsehuman_test_x', arr=horsehuman_test[0][0])  # x_test
# np.save(path + 'horsehuman_test_y', arr=horsehuman_test[0][1])  # y_test

#################################### Load ####################################
#1. load data
x_train = np.load(path + 'horsehuman_train_x.npy')  # (500, 200, 200, 3)
y_train = np.load(path + 'horsehuman_train_y.npy')  # (500,)   
x_test = np.load(path + 'horsehuman_test_x.npy')  # (205, 200, 200, 3)
y_test = np.load(path + 'horsehuman_test_y.npy')  # (205,)

ic(x_train.shape, y_train.shape, x_test.shape, y_test.shape) 

#2. modeling(VGG19, Xception)
model_list = [VGG19(weights='imagenet', include_top=False, input_shape=(200,200,3), pooling='max', classifier_activation='sigmoid'),
              Xception(weights='imagenet', include_top=False, input_shape=(200,200,3), pooling='max', classifier_activation='sigmoid')]
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
    es = EarlyStopping(monitor='val_acc', patience=10, mode='max', verbose=1, restore_best_weights=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='acc')
    
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
Time : 22.903
loss : 0.1644
Acc : 0.95

Model : xception
Time : 22.3753
loss : 0.0045
Acc : 1.0
'''