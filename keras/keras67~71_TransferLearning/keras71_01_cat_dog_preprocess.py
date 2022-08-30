from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception,VGG19
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Flatten,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np, time, warnings, os
from glob import glob
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from icecream import ic
# warnings.filterwarnings(action='ignore')

#1. load images 
def load_img_to_numpy(path):
    path = path
    images = []
    labels = []
    
    for filename in glob(path +"*"):
        for img in glob(filename + "/*.jpg"):
            an_img = Image.open(img).convert('RGB').resize((100, 100))  # read img
            img_array = np.array(an_img)  # img to array
            images.append(img_array)  # append array to training_images 
            label = filename.split('\\')[-1]  # get label
            labels.append(label)  # append label
    
    # list -> np.array        
    images = np.array(images)  
    labels = np.array(labels) 

    # LabelEncoder
    le = LabelEncoder()
    labels= le.fit_transform(labels)
    # labels = labels.reshape(-1,1)
    return images, labels


train_path = 'keras/data/images/cat_or_dog/train/'
test_path = 'keras/data/images/cat_or_dog/test/'

x_train, y_train = load_img_to_numpy(train_path)
x_test, y_test = load_img_to_numpy(test_path)

#2. modeling(VGG19, Xception)
model_list = [VGG19(weights='imagenet', include_top=False, input_shape=(100,100,3),pooling='avg', classifier_activation='sigmoid'),
              Xception(weights='imagenet', include_top=False, input_shape=(100,100,3),pooling='avg', classifier_activation='sigmoid')]
for model in model_list:
    TL_model = model
    # preprocess
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test) 
    model = Sequential()
    model.add(TL_model)    
    # classifier 
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #3. compile, train
    optimizer = Adam(learning_rate=0.0001)  # 1e-4     
    lr=ReduceLROnPlateau(monitor='val_acc', patience=3, mode='max', factor=0.1, min_lr=0.00001, verbose=False)
    es = EarlyStopping(monitor='val_acc', patience=15, mode='max', verbose=1, restore_best_weights=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='acc')
    
    start = time.time()
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.2, callbacks=[lr, es], verbose=1)
    end = time.time()
    
    #4. evaluate
    loss, acc = model.evaluate(x_test, y_test, batch_size=32, verbose=True)
    print(f"Model : {TL_model.name}")
    print(f"Time : {round(end-start, 4)}")
    print(f"loss : {round(loss, 4)}")
    print(f"Acc : {round(acc, 4)}")
    
# Error: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info. This isn't available when running in Eager mode.
# -----> GPU 메모리 부족, batch_size 100 -> 32로 수정   

'''
Model : vgg19
Time : 23.7472
loss : 0.7036
Acc : 0.6337

Model : xception
Time : 22.4231
loss : 0.5288
Acc : 0.7673
'''