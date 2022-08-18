from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions     
import numpy as np
# input에 전처리해준다.  incode 암호화, decode 복호화

model = ResNet50(weights='imagenet',pooling='avg')

img_path = '../_data/cat_dog.jpg'

img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
print("=========================image.img_to_array(img) ======================")
print(x, '\n', x.shape)

x = np.expand_dims(x, axis=0)
print("=========================np.expand_dims(x, axis=0) ======================")
print(x, '\n', x.shape)

x = preprocess_input(x)
print("=========================preprocess_input ======================")
print(x, '\n', x.shape)

preds = model.predict(x)

print(preds, '\t', preds.shape)

print('결과는', decode_predictions(preds, top=5))   # label이름을 argmax먹여서 되돌려준다