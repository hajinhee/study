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
# 이 옵션들이 되게 중요함. 각 이미지 타입들마다 그 이미지들의 특성이 있기때문에 변환특성을 잘 보고 각 데이터에 맞게 적절한 비율로 커스터마이즈 해줘야함. ****

augment_size = 100
# print(x_train[0].shape)                                                                 # (28, 28)
# print(x_train[0].reshape(28*28).shape)                                                  # (784,)
# print(np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1).shape)        # (100, 28, 28, 1)

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augment_size).reshape(-1,28,28,1),   # 28*28대신 -1도 가능. np.tile 반복한다. 784,라는 데이터를 아큐먼트 사이즈만큼 반복하겠다.
                                                                            # x_train의 제일 첫번째 사진[0]을 784로 쫙펴서 100번 반복한다 여기서 100장의 사진이 완성된다.
    np.zeros(augment_size),          # 위에서 1장의사진으로 만들어진 100개의 같은 사진에 y값 100개를 전부다 0으로 넣어주겠다.
    batch_size=augment_size,         # 100개로 쫙 펴진걸 100개로 묶어줌 -> 1개의 큰 []안에 100개의 사진을 다 넣겠다.   
    shuffle=False                    # 셔플꺼서 다 똑같은 형태로 변환되지않나? -> 추후에 확인해봐야함.
).next()                             # 형태변환을 시켜줌. 마치 .values로 판다스에서 넘파이로 바꾸듯이 데이터의 자료형을 이터러블?한것에서 배열형태로 해줌. 자료형태에 따라 리스트 튜플 넘파이 등
                                     # .next() 메소드는 선택한 요소의 바로 다음에 위치한 형제 요소를 선택한다. 클래스를 가진 요소의 바로 다음 형제 요소 하나를 선택하여, 해당 요소의 CSS 스타일을 변경한다.

#print(len(x_data))
print(type(x_data))                         # <class 'tuple'>
# print(x_data[0].shape, x_data[1].shape)     # (100, 28, 28, 1) (100,)   튜플로 묶인 세트의 0과1이므로 곧 x와 y값 의미.

# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,7))
# for i in range(49):
#     plt.subplot(8,8,i+1)
#     plt.axis('off')
#     plt.imshow(x_data[0][i], cmap='gray')
# plt.show()                                  # 100배로 불린 사진들을 49개만 판에 출력해보겠다.

# 복습 다시 필요함.