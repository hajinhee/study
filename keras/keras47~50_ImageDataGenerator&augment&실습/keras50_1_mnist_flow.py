# 훈련데이터 10만개로 증폭
# 기존 모델과 비교
# flow방식 사용
# save_dir도 _temp에 넣고
# 증폭데이터는 temp에 저장후 훈련 끝난후 결과 보고 삭제
# 1~3은 증폭10만개 
# 1번파일은 하는김에 제대로 하게 좀 자세하게 훝어보고 가자.
# 그림도 찍어서 데이터 눈으로 확인해보자.

from typing import Tuple
from tensorflow.keras import datasets
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(action='ignore')

#1.데이터 로드 및 전처리.

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()
#print(x_train.shape,y_train.shape)     (60000, 28, 28) (60000,)    6만장의 사진과 그에 대한 답입니다~

#그림을 직접 한번 보고 가겠습니다.
# plt.figure(figsize=(10,10))
# for i in range(30):
#     plt.subplot(8,8,i+1)
#     plt.imshow(x_test[i])
# plt.show()

#이 데이터는 신발,원피스,티셔츠,바지 등등 10가지의 의류?의 img를 담고 있습니다. -> 사실 아니었다. 넘파이 배열을 담고 있다.
#이걸 plt기능을 이용해서 x_train을 뿌려서 눈으로 확인해보니까 그게 img의 모습이었던거지 우리는 한번도 이미지 그 자체를 로드한적이 없었습니다.
#type(x_train뭐 기타등등) 확인해보면 numpy라고 나오지 image라고 나오지 않아요. 왜냐? 이미지 제너레이터를 통해서 불러왔기때문에~
#이미 불러오고 난 후엔 넘파이로 불러온 뒤라서 넘파이였고, 변환해준것도 넘파이로 만들어서 배열끼리 더해주면 (이걸 증폭이라고 함) 10만개의 x_train을 만들수 있습니다.

#아쉽게도 제공 데이터가 numpy라서 0이 뭔지 1이뭔지...9가 뭔지는 확인할 수 없습니다. 너무 아쉽네요~
#눈대중으로 뭐가 뭔지 직접 확인해서 기록해야 할것 같지만 귀찮아서 패스하도록 하겠습니다~

#이미지데이터제눠뤠이션 하기전에 옵션들을 공부하고 가겠습니다. 너무 많아서 따로 파일 만들겠습니다.
#ImageDataGenerator_options.txt 참고. 

# test파일 확인해보고 쌤한테 질문도 해봤는데 tensorflow 자격증 시험가더라도 test파일 막 그렇게 개판으로 주지는 않아서 
# 뒤집고 기울이고 하는 극단적인 옵션은 굳이 안써도 될거 같습니다.

train_augment_datagen = ImageDataGenerator(    
    #rescale=1./255,        #변환 시킨 후 train데이터에 얹어서(증폭)한 후 다시 이미지제너레이터할거기때문에 scale은 노노
    horizontal_flip=True,   #좌우반전 
    rotation_range=3,       #회전범위지정
    width_shift_range=0.3,  #수평,좌우 이동 범위
    height_shift_range=0.3, #수직,상하 이동 범위 
    zoom_range=(0.3),        #원래 lower upper값이 따로 들어가는데 한 값넣어주면 알아서 1 - 0.3, 1 + 0.3해서 잡아줌. 
    fill_mode='nearest',                    
       
)
all_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)


'''
flow로 데이터 x,y따로 받아와서 이미지제너레이터 해주는 작업은 여기 단계가 아니고 그 이후.       
원본6만개 + 증폭변환4만개 할꺼고 10만개 완성되면 그걸 train&val&test 나눠줄거니까.
'''

# 지금부터 변환을 해보겠습니다. 

augment_size = 100000 - x_train.shape[0]                        # 증폭할 데이터의 개수 지정 총 10만개해주기 위해서 40000개 지정.
randidx = np.random.randint(x_train.shape[0],size=augment_size) # 랜덤하게 x_train에서 40000개를 뽑음 -> 각 1장마다 변환을 할거임.
#print(len(randidx),randidx[:10])    #40000 [35782 42200 29646  4542 12414 43479 44382 15719 56537 21100]
#랜덤한 값 40000개가 크고 작음 상관없이 들어가있는걸 볼 수 있다.


x_augmented = x_train[randidx].copy()      
y_augmented = y_train[randidx].copy() 
#print(len(x_augmented),len(y_augmented))   # 40000 40000
#x_augmented와 y~에 x_train에서 뽑아낸 40000개의 값들을 .copy()로 복사해서 메모리에 한번 실어준 후 값들을 똑같이 저장한다.
#print(type(x_augmented))  <class 'numpy.ndarray'>

x_augmented = x_augmented.reshape(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2],1)
x_train = x_train.reshape(60000,28,28,1)            # 변환된 값 원래 트레인에 더해주기위해 변환
x_test = x_test.reshape(x_test.shape[0],28,28,1)    # train과 똑같은 상태로변환
# x를 flow로 받아서 이미지제네레이터 하기 위해 4차원으로 형태 변환. (samples, height, width, channels)

# fit,flow,flow_from_dataframe,flow_from_directory 4개가 있는데 
# flow는 이미지를 x y로 따로 받아서 묶어서 저장하고 dataframe은 이미지 자체를 받아서? directory는 폴더째로 받아서 해주는거 같다. fit은 개념도 안잡힌다.

#x_augmented에 이미지제너레이터써서 변환시키기전에 확인해보고 그 후에 확인해보면 차이를 볼 수 있다. -> 변환전의 값을 before로 담아준다.
before_x_augmented = x_augmented.copy()

x_augmented = train_augment_datagen.flow(
    x_augmented, y_augmented,
    batch_size=augment_size, shuffle=False #, seed=66,     #shuffle했더니 변환전과 후의 사진들의 순서가 뒤죽박죽 되어있어서 변환을 측정할수가없었다.
    # shuffle의 의미 : x,y각각 가져와서 변환후 짝 지어줄때 train_test_split했을때처럼 그 index상의 순서를 셔플해준다.변환값들은 자기가 알아서 임의로 뽑아서 변환해준다.
).next()[0]     
#next()와 [0] 을 해준이유 이미지제너레이터한 결과물을 넘파이형태로 x값만 따와서 저장하기위해.
#변환전의 x_train데이터는 x값만 있는 형태기때문에 갑자기 xy묶어서 변환하면 변환 전후의 type이 바뀌고 원래의 train데이터도 xy로 합쳐준 후
#변환된 xy를 합춰줘야 한다. -> 이 방법이 가능하긴한가?
#batch_size = augment_size 한 이유. 배치사이즈를 데이터 전체개수만큼 줘서 1개로 묶기위해. 그 후 x값만 따와서 리스트로 해주면 각 x를 하나씩 카운트할수있다.
#이터러블?이터레이터?한 자료형 -> 넘파이로 바뀐다. -> x,y를 각각 넣는 fit연산을 할 수 있다.(x_train과x_augment)를 더할수있다. 이 작업을 증폭이라한다.

#train에 변환된 argument를 넣어주기전 변환이 잘 되었나 그림을 통해 확인해보자. -> 변환후의 값을 aftef로 담아준다.
after_x_augmented = x_augmented.copy()

'''
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.subplot(2,10,i+1)
    plt.axis('off')
    plt.imshow(before_x_augmented[i])
    plt.imshow(after_x_augmented[i])
plt.show()       
plt.figure(figsize=(10,10))

for i in range(10):
    plt.subplot(8,8,i+1)
    plt.axis('off')
    plt.imshow(after_x_augmented[i])    
plt.show()   
'''

#하... 한번에 2줄로 깔끔하게 변환 전후 출력되게 하고싶은데 숙제때문에 일단 접어두도록 하겠습니다...

real_x_train = np.concatenate((x_train, x_augmented))   
real_y_train = np.concatenate((y_train, y_augmented))
#print(len(real_x_train),type(real_x_train))        #100000 <class 'numpy.ndarray'>
#print(len(real_y_train),type(real_y_train))        #100000 <class 'numpy.ndarray'>

#이제야 비로소 진짜 x_train과 y_train이 10만개씩 만들어졌으므로 이미지제너레이터를 새로해서 작업해보도록하겠습니다.

xy_train_train = all_datagen.flow(
    real_x_train,real_y_train,
    batch_size=100,shuffle=True,seed=66,
    subset='training'
)#.next()[0] 
xy_train_val = all_datagen.flow(
    real_x_train,real_y_train,
    batch_size=100,shuffle=True,seed=66,
    subset='validation'
)#.next()[0]
xy_test = all_datagen.flow(
    x_test,y_test,
    batch_size=100
)#.next()[0]
# all_datagen에 validation_split써놓긴 했지만 subset으로 딱딱 명시해줘야 작동하지 굳이 명시 안하면 스킵하고 잘 작동한다.
#print(len(xy_train_train),len(xy_train_val),len(xy_test)) #80000 20000 10000
#flow 그냥 하는것과 .next() .next()[0]의 차이
#그냥 할 경우
# 80000 길이
# <class 'keras.preprocessing.image.NumpyArrayIterator'>
# 2
# <class 'tuple'>
# 1
# <class 'numpy.ndarray'>

#.next()
# 2 길이
# <class 'tuple'>
# 1
# <class 'numpy.ndarray'>
# 28
# <class 'numpy.ndarray'>

#.next()[0]     [0]을 쓰면 x값만 가져오겠다는 의미 
# 1 길이 
# <class 'numpy.ndarray'>
# 28
# <class 'numpy.ndarray'>
# 28
# <class 'numpy.ndarray'>
# keras49_1_flow에 적은내용이 그대로 나타난다. next() 메소드는 선택한 요소의 바로 다음에 위치한 형제 요소를 선택한다.
# 클래스를 가진 요소의 바로 다음 형제 요소 하나를 선택하여, 해당 요소의 CSS 스타일을 변경한다.


#2. 모델링

model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2), input_shape=(x_train.shape[1],x_train.shape[2],1)))  
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
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy']) 

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit_generator(xy_train_train,epochs=1,steps_per_epoch=len(xy_train_train)//2,validation_data=xy_train_val, validation_steps=len(xy_train_val),callbacks=[es])

#4. 평가 예측
loss = model.evaluate_generator(xy_test, steps=len(xy_test))
print('loss : ', loss[0])
print('accuracy : ', loss[1])

y_pred = model.predict_generator(x_test)

y_pred_int = np.argmax(y_pred,axis=1)        

acc = accuracy_score(y_test,y_pred_int)

print('acc_scroe : ',acc)

'''
# 현재        증폭시키기전 값.
# loss :      0.31123247742652893       0.3209112584590912      0.32448408007621765
# accuracy :  0.8934000134468079        0.8866000175476074      0.888700008392334

# train 60000 -> 100000만으로 증폭시킨 후 .
# loss :     0.3264023959636688 
# accuracy : 0.891700029373169 
# acc_score: 0.6707  왜 차이나지?                         
'''