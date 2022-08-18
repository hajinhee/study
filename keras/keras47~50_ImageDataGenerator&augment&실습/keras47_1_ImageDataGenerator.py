import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator     # 대문자니까 class

train_datagen = ImageDataGenerator(     # 함수선언 및 정의 
    rescale=1./255,                     # Minmax scale해주겠다. 이미지는 최소 최대값이 0~255사이이므로 이 작업해주는게 곧 minmax작업과 같음
    horizontal_flip=True,               # 상하반전  mnist같은건 상하반전하면 안된다. 잘 생각하고 쓰기.
    vertical_flip=True,                 # 좌우반전                                  
    width_shift_range=0.1,              # 이미지를 ...? 이동?
    height_shift_range=0.1,             # 이미지를 0.1만큼 이동시킴? 비교하기 위해서?
    rotation_range=5,                   # 이미지를 회전시켜줌
    zoom_range=1.2,                     # 확대
    shear_range=0.7,                    #
    fill_mode='nearest'                 # 이미자를 이동시키거나 뭐 했을때 빈곳을 근처의 값으로 채우겠다.
    # 기타 등등 파라미터 더 있는데 잘 정리하기.
    # 이미지를 제자리에서 이동시키면서,상하좌우반전시키면서,여러가지 형태의 이미지를 모두 학습시킴으로써 어떤형태의 이미지라도 판단할수 있게함.
)

test_datagen = ImageDataGenerator(
    rescale=1./255                      # 테스트 데이터는 증폭이나 변조를 하지않는다. <-- 평가할때는 원본으로 평가해야 정확한 검증할 수 있다.
)

# D:\_data\image\brain  

xy_train = train_datagen.flow_from_directory(       # 경로설정. 분류형식의 이미지 받으면 폴더형식에 맞춰서 세팅.
    '../_data/image/brain/train/',
    target_size = (150, 150),                       # 데이터의 사이즈 크기를 넣는게 아니라 내가 원하는 데이터 사이즈를 지정하면
                                                    # 원래 데이터를 여기사이즈에 맞게 맞춰서 작업한다. 그래도 적당히는 값 맞춰서 해줘야한다. 
    batch_size=5,                                   # 훈련시킬때 batch_size를 여기서 지정해준다.
    class_mode='binary',                            # 이게 뭘까용
    shuffle=True,    
)   # Found 160 images belonging to 2 classes.      # train에 대한 데이터가 수치화 되어 빠진다.
# 160개의 train사진을 batch사이즈 단위로 묶어서 수치화해서 저장한다. batch가 5니까 160/5해서 32개의 묶음으로 나온다.

xy_test = test_datagen.flow_from_directory(         # 정의단계에서 세팅을 다 하고 flow~로 데이터 가져와서 xy분류해준다.
    '../_data/image/brain/test/',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary',                            # shuffle 안해도 상관없다.
)   # Found 120 images belonging to 2 classes.

# 파이토치의 데이터로더 및 텐서플로의 이미지제너레이터 등등 다 이런형식으로 저장되어있고 불러온다.

#print(xy_train)
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x000001892A484F70>

# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets)           

#print(xy_train[0])      # y값이 y의 batch사이즈 개수만큼 나온다.
#print(xy_train[0][0])    # xy_train의 첫번째 묶음의 첫번째 = x
#print(xy_train[0][1])    # xy_train의 첫번째 묶음의 두번째 = y
#print(xy_train[0][0].shape, xy_train[0][1].shape)     # (5, 150, 150, 3)  (5,)   5개씩 묶었으므로 5, 150,150원래 사이즈 흑백처럼보이지만 컬러사진이었다 그래서 채널 3

#print(type(xy_train))  #<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
#print(type(xy_train[0]))     # <class 'tuple'>
#print(type(xy_train[0][0]))  # <class 'numpy.ndarray'>
#print(type(xy_train[0][1]))  # <class 'numpy.ndarray'>

