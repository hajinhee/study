from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from icecream import ic
import matplotlib.pyplot as plt
import seaborn as sns


#1. 데이터 로드
datasets = load_boston()
x = datasets.data  # (506, 13)      
y = datasets.target  # (506, )       

print(datasets.feature_names) # 컬럼명
'''
['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
'''
print(datasets.DESCR) # 데이터셋 및 컬럼에 대한 설명 
'''
**Data Set Characteristics:**
:Number of Instances: 506
:Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
:Attribute Information (in order):
    - CRIM     per capita crime rate by town
    - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    - INDUS    proportion of non-retail business acres per town
    - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    - NOX      nitric oxides concentration (parts per 10 million)
    - RM       average number of rooms per dwelling
    - AGE      proportion of owner-occupied units built prior to 1940
    - DIS      weighted distances to five Boston employment centres
    - RAD      index of accessibility to radial highways
    - TAX      full-value property-tax rate per $10,000
    - PTRATIO  pupil-teacher ratio by town
    - B        1000(Bk - 0.63)^2 where Bk is the proportion of black people by town
    - LSTAT    % lower status of the population
    - MEDV     Median value of owner-occupied homes in $1000's
'''

#2. pandas dataframe으로 전환 후 컬럼명 삽입
xx = pd.DataFrame(x, columns=datasets.feature_names)  # pandas로 변환 후 pandas에서 제공하는 index와 columns정보를 확인할 수 있다.   
ic(xx.columns) 
'''
['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
'''
ic(xx.ndim)  # 2d
ic(xx.shape)  # (506, 13)


#3. 컬럼 추가
xx['PRICE'] = y  # y값을 xx데이터에 'PRICE'라는 이름의 칼럼으로 추가한다.
ic(xx.columns) 
'''
['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE']
'''

#4. 상관계수 확인
ic(xx.corrwith(xx['PRICE']))  # PRICE와 어떤 열이 제일 상관관계가 적은지 확인
'''
CRIM      -0.388305
ZN         0.360445
INDUS     -0.483725
CHAS       0.175260
NOX       -0.427321
RM         0.695360
AGE       -0.376955
DIS        0.249929
RAD       -0.381626
TAX       -0.468536
PTRATIO   -0.507787
B          0.333461
LSTAT     -0.737663
PRICE      1.000000
'''
plt.figure(figsize=(10, 10))
sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)
# plt.show()


#5. 불필요한 컬럼 삭제
xx.drop(['CHAS', 'PRICE'], axis=1, inplace=True)    
ic(xx.columns)     


#6. numpy로 변환
xx = xx.values  # 또는 xx = xx.to_numpy()  
'''
[[6.3200e-03, 1.8000e+01, 2.3100e+00, ..., 1.5300e+01, 3.9690e+02, 4.9800e+00],
[2.7310e-02, 0.0000e+00, 7.0700e+00, ..., 1.7800e+01, 3.9690e+02, 9.1400e+00],
[2.7290e-02, 0.0000e+00, 7.0700e+00, ..., 1.7800e+01, 3.9283e+02, 4.0300e+00],
'''
ic(xx.ndim)  # 2d
ic(xx.shape)  # (506, 12)


#7. 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(xx, y, train_size=0.9, shuffle=True, random_state=42)
# print(x_train.shape,y_train.shape) (455, 12) (455,)
# print(x_test.shape,y_test.shape) (51, 12) (51,)


#8. 스케일링
scaler = StandardScaler()  # 스케일러는 2차원 데이터일 때만 사용이 가능하다. 스케일링 후 바로 4차원 데이터로 변환한다.
x_train = scaler.fit_transform(x_train).reshape(len(x_train), 3, 4, 1)  # (455, 3, 4, 1)
x_test = scaler.transform(x_test).reshape(len(x_test),3,4,1)  # (51, 3, 4, 1)
 

#9.모델링
model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),strides=1,padding='same', input_shape=(3,4,1), activation='relu'))                                                                             # 1,1,10
model.add(Conv2D(15,kernel_size=(1,2), strides=1, padding='valid', activation='relu'))                     
model.add(Conv2D(20,kernel_size=(2,2), strides=1, padding='valid', activation='relu'))             
model.add(MaxPooling2D(2,2))                                                                            
model.add(Flatten())       
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Dropout(0.2))
model.add(Dense(1))


ti = time.time()
kr = time.localtime(ti)
krtime = time.strftime('%m-%d-%X',kr).replace(":", "_")

#10. 모델 컴파일, 훈련, 저장
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1, baseline=None, restore_best_weights=True)
# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras35_1_boston{krtime}.hdf5')
model.fit(x_train, y_train, epochs=500, batch_size=10, validation_split=0.111111, callbacks=[es])
model.save(f"./_save/keras35_1_boston{krtime}.h5")


#11. 평가
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)


#12. 예측
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)


'''
[loss] :  7.407820224761963
[r2스코어] :  0.8813500235703937
'''
