from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions     
import numpy as np
from icecream import ic

# pretrained model 
model = ResNet50(weights='imagenet', pooling='avg')

# load image
img_path = 'keras/data/images/cat_dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))  # 단일 이미지를 로드하면 하나의 이미지 모양인 (size1, size2, channels) 를 얻게 된다.

# image to array
x = image.img_to_array(img) 
ic(x.shape, x)  # (224, 224, 3)

# expand dims
x = np.expand_dims(x, axis=0)  # 이미지 배치를 생성하려면 추가 차원이 필요하다. (samples, size1, size2, channels)
ic(x.shape, x)  # (1, 224, 224, 3)

# preprocess input
x = preprocess_input(x)  # preprocess_input() 함수는 모델에 필요한 형식에 이미지를 적절하게 맞추기 위한 것이다.
ic(x.shape, x)

# predict
y_pred = model.predict(x)
ic(y_pred.shape, y_pred)  # (1, 1000)

print('결과는', decode_predictions(y_pred, top=5))  # decode_predictions()를 통해 가능성이 높은 상위 클래스와 확률을 리스트 형식으로 반환한다.

'''
결과는 [[('n02124075', 'Egyptian_cat', 0.37234), 
        ('n02123159', 'tiger_cat', 0.35174936), 
        ('n02123045', 'tabby', 0.10951182), 
        ('n02123597', 'Siamese_cat', 0.021259649), 
        ('n02127052', 'lynx', 0.016386898)]]
'''