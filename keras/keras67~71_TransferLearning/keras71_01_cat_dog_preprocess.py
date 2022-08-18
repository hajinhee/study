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
from PIL import Image

# warnings.filterwarnings(action='ignore')

def load_img_to_numpy(path):
    
    path = path
    images = []
    labels = []
    
    for filename in glob(path +"*"):
        for img in glob(filename + "/*.jpg"):
            an_img = Image.open(img).convert('RGB').resize((100,100)) #read img
            img_array = np.array(an_img) #img to array
            images.append(img_array) #append array to training_images 
            label = filename.split('\\')[-1] #get label
            labels.append(label) #append label
            
    images = np.array(images)
    labels = np.array(labels)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels= le.fit_transform(labels)
    # labels = labels.reshape(-1,1)
    
    return images, labels

train_path = 'D:\_data\image_classification\cat_dog/training_set/'
test_path = 'D:\_data\image_classification\cat_dog/test_set/'

x_train,y_train = load_img_to_numpy(train_path)
x_test,y_test = load_img_to_numpy(test_path)

model_list = [VGG19(weights='imagenet', include_top=False, input_shape=(100,100,3),pooling='avg',classifier_activation='sigmoid'),
              Xception(weights='imagenet', include_top=False, input_shape=(100,100,3),pooling='avg',classifier_activation='sigmoid')]
            # pooling의 역할. 전이학습 모델의 마지막레이어를 flatten넣을지 globalavarge 넣을지 고를 수 있다...
for model in model_list:
    print(f"모델명 : {model.name}")
    TL_model = model
    x_train = preprocess_input(x_train)   #  요기 전처리 과정 추가.
    x_test = preprocess_input(x_test) 
    model = Sequential()
    model.add(TL_model)    
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    optimizer = Adam(learning_rate=0.0001)  # 1e-4     
    lr=ReduceLROnPlateau(monitor= "val_acc", patience = 3, mode='max',factor = 0.1, min_lr=0.00001,verbose=False)
    es = EarlyStopping(monitor ="val_acc", patience=15, mode='max',verbose=1,restore_best_weights=True)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='acc')
    
    start = time.time()
    model.fit(x_train,y_train,batch_size=50,epochs=1000,validation_split=0.2,callbacks=[lr,es], verbose=1)
    end = time.time()
    
    loss, Acc = model.evaluate(x_test,y_test,batch_size=50,verbose=False)
    
    print(f"Time : {round(end - start,4)}")
    print(f"loss : {round(loss,4)}")
    print(f"Acc : {round(Acc,4)}")

'''
모델명 : vgg19
Time : 365.7781
loss : 0.2059
Acc : 0.9224

모델명 : xception
Time : 791.78
loss : 0.2327
Acc : 0.9417
'''