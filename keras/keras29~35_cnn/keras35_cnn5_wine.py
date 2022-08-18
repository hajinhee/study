from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import inspect, os
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

#1. 데이터 
datasets = load_wine()
x = datasets.data  # (178, 13)           
y = datasets.target  # (178,)        
ic(np.unique(y, return_counts=True))  # (array([0, 1, 2]), array([59, 71, 48], dtype=int64)) --> 다중분류 

ic(datasets.DESCR)  # 데이터셋 및 컬럼에 대한 설명 
ic(datasets.feature_names)  # 컬럼명
'''
['alcohol',
'malic_acid',
'ash',
'alcalinity_of_ash',
'magnesium',
'total_phenols',
'flavanoids',
'nonflavanoid_phenols',
'proanthocyanins',
'color_intensity',
'hue',
'od280/od315_of_diluted_wines',
'proline']
'''
ic(type(x), x)    
'''
<class 'numpy.ndarray'>
[[1.423e+01 1.710e+00 2.430e+00 ... 1.040e+00 3.920e+00 1.065e+03]
 [1.320e+01 1.780e+00 2.140e+00 ... 1.050e+00 3.400e+00 1.050e+03]
 [1.316e+01 2.360e+00 2.670e+00 ... 1.030e+00 3.170e+00 1.185e+03]
'''  

# numpy -> pandas로 변환 후 컬럼명 삽입
xx = pd.DataFrame(x, columns=datasets.feature_names)  # pandas로 변환 후 pandas에서 제공하는 index와 columns정보를 확인할 수 있다.   
ic(type(xx), xx)      
'''
<class 'pandas.core.frame.DataFrame'>
    alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  od280/od315_of_diluted_wines  proline   
0      14.23        1.71  2.43               15.6      127.0           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92   1065.0   
1      13.20        1.78  2.14               11.2      100.0           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40   1050.0   
2      13.16        2.36  2.67               18.6      101.0           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17   1185.0   
3      14.37        1.95  2.50               16.8      113.0           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45   1480.0   
4      13.24        2.59  2.87               21.0      118.0           2.80        2.69                  0.39             1.82             4.32  1.04                          2.93    735.0 
'''

# 데이터 상관관계를 알기 위해 y 데이터를 xx데이터에 'PRICE'라는 이름의 컬럼으로 추가 --> 컬럼 추가            
xx['price'] = y        
ic(xx)         
'''
    alcohol  malic_acid   ash  alcalinity_of_ash  magnesium  total_phenols  ...  proanthocyanins  color_intensity   hue  od280/od315_of_diluted_wines  proline  price
0      14.23        1.71  2.43               15.6      127.0           2.80  ...             2.29             5.64  1.04                          3.92   1065.0      0
1      13.20        1.78  2.14               11.2      100.0           2.65  ...             1.28             4.38  1.05                          3.40   1050.0      0
2      13.16        2.36  2.67               18.6      101.0           2.80  ...             2.81             5.68  1.03                          3.17   1185.0      0
3      14.37        1.95  2.50               16.8      113.0           3.85  ...             2.18             7.80  0.86                          3.45   1480.0      0
4      13.24        2.59  2.87               21.0      118.0           2.80  ...             1.82             4.32  1.04                          2.93    735.0      0
'''

# xx['price'] 과의 상관관계 확인
ic(xx.corrwith(xx['price']))     
'''
alcohol                        -0.328222
malic_acid                      0.437776
ash                            -0.049643
alcalinity_of_ash               0.517859
magnesium                      -0.209179
total_phenols                  -0.719163
flavanoids                     -0.847498
nonflavanoid_phenols            0.489109
proanthocyanins                -0.499130
color_intensity                 0.265668
hue                            -0.617369
od280/od315_of_diluted_wines   -0.788230
proline                        -0.633717
price                           1.000000
'''

# 상관관계 분석도
plt.figure(figsize=(10, 8))
heat_table = xx.corr()
mask = np.zeros_like(heat_table)
mask[np.triu_indices_from(mask)] = True
heatmap_ax = sns.heatmap(heat_table, square=True, cbar=True, annot=True, mask = mask, cmap='coolwarm', vmin=-1, vmax=1)
heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), fontsize=10, rotation=90)
heatmap_ax.set_yticklabels(heatmap_ax.get_yticklabels(), fontsize=10)
plt.title('correlation between features', fontsize=20)
# plt.show()

# 불필요한 컬럼 삭제
xx.drop(['ash','price'], axis=1, inplace=True)   
ic(xx)     
'''
    alcohol  malic_acid  alcalinity_of_ash  magnesium  total_phenols  flavanoids  nonflavanoid_phenols  proanthocyanins  color_intensity   hue  od280/od315_of_diluted_wines  proline
0      14.23        1.71               15.6      127.0           2.80        3.06                  0.28             2.29             5.64  1.04                          3.92   1065.0 
1      13.20        1.78               11.2      100.0           2.65        2.76                  0.26             1.28             4.38  1.05                          3.40   1050.0
2      13.16        2.36               18.6      101.0           2.80        3.24                  0.30             2.81             5.68  1.03                          3.17   1185.0 
3      14.37        1.95               16.8      113.0           3.85        3.49                  0.24             2.18             7.80  0.86                          3.45   1480.0 
4      13.24        2.59               21.0      118.0           2.80        2.69                  0.39             1.82             4.32  1.04                          2.93    735.0 
..       ...         ...                ...        ...            ...         ...                   ...              ...              ...   ...                           ...      ...
173    13.71        5.65               20.5       95.0           1.68        0.61                  0.52             1.06             7.70  0.64                          1.74    740.0 
174    13.40        3.91               23.0      102.0           1.80        0.75                  0.43             1.41             7.30  0.70                          1.56    750.0 
175    13.27        4.28               20.0      120.0           1.59        0.69                  0.43             1.35            10.20  0.59                          1.56    835.0
176    13.17        2.59               20.0      120.0           1.65        0.68                  0.53             1.46             9.30  0.60                          1.62    840.0 
177    14.13        4.10               24.5       96.0           2.05        0.76                  0.56             1.35             9.20  0.61                          1.60    560.0 
'''

# 다시 numpy로 변환
xx = xx.to_numpy()  # 또는 xx = xx.values     

x_train, x_test, y_train, y_test = train_test_split(xx, y, train_size=0.9, shuffle=True, random_state=49)

scaler = MinMaxScaler()   
x_train = scaler.fit_transform(x_train).reshape(len(x_train),3,4,1)  # (160, 3, 4, 1)
x_test = scaler.transform(x_test).reshape(len(x_test),3,4,1)  # (18, 3, 4, 1)
y_train = to_categorical(y_train)  # 원핫인코딩 --> (160, 3)
y_test = to_categorical(y_test)  


#2.모델링
model = Sequential()
model.add(Conv2D(4, kernel_size=(2, 2), strides=1, padding='same', input_shape=(3, 4, 1), activation='relu'))                                                                    # 1,1,10
model.add(Conv2D(4, kernel_size=(2, 3), strides=1, padding='valid', activation='relu'))                                                                                     # 1,1,10
model.add(MaxPooling2D(2, 2))                                                                                                   
model.add(Flatten())       
model.add(Dense(40))
model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, baseline=None, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras35_2_diabetes{krtime}.hdf5')
model.fit(x_train, y_train, epochs=100, batch_size=3, validation_split=0.111111, callbacks=[es])

#4. 평가 예측
loss = model.evaluate(x_test ,y_test)
print('[loss] : ', loss[0])
print('[accuracy] : ', loss[1])

acc= str(round(loss[1], 4))
model.save(f"./_save/keras35_5_wine_acc_Min_{acc}.h5")


'''
[loss] :  0.06036273017525673
[accuracy] :  1.0
'''
