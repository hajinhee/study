from pandas.core.frame import DataFrame
from scipy.sparse import data
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPooling2D
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pandas import get_dummies,DataFrame
import inspect, os
import math


#1 데이터
datasets = fetch_covtype()
x = datasets.data  # (581012, 54)         
y = datasets.target  # (581012,)       


#print(np.unique(y, return_counts=True))    # label값 7개 1번데이터만 양이 압도적으로 많다.

y = get_dummies(y)  # 확인해보면 colunms명이 1~7까지 잘 들어가있다.

'''
# 이 데이터는 columns가 54개라서 한번 까봐야속이 시원할거같다.

# numpy pandas로 변환후 pandas의 제공기능인 index정보와 columns정보를 확인할수있다.
#xx = pd.DataFrame(x, columns=datasets.feature_names)    # x가 pandas로 바껴서 xx에 저장, columns를 칼럼명이 나오게 지정해준다.

#print(type(xx))         # pandas.core.frame.DataFrame
#print(xx)               # 잘 되었나 확인. index와 colmuns의 이름이 나옴.

#print(xx.corr())        # 칼럼들의 서로서로의 상관관게를 수치로 확인할 수 있다.    절대값클수록 양 or 음의 상관관계 0에 가까울수록 서로 영향 없음

#xx['y~~'] = y         # xx의 데이터셋에 y값을 price라는 이름의 칼럼으로 추가한다. 원본데이터는 그대로있다.    열 추가하는 방법.

#print(xx)              # price열이 추가되어 있는 것 확인.

#print(xx.corr())      # price와 어떤 열이 제일 상관관계가 적은지 확인.

#xx.corr()['y~~']    #<class 'pandas.core.series.Series'>   pandas의 y값을 포함한 xx데이터 상관관계 표의 y~~열

# for i in xx.corr()['y~~']:
#     if i > 0 and i < 0.001:           # xx데이터의 상관관계파일속 y칼럼 상관값에서 0.001이하의 값들은 그 값들의 colum이름을 리스트에 담고 반환해주면 그 리스트를 바로 삭제하면 될텐데.
#            d.append(i.index.values)

# print(d)                                            

#########################################################
# import matplotlib.pyplot as plt
# import seaborn as sns   # 조금 더 이쁘게 만들게 도와줌.
# plt.figure(figsize=(10,10))
# sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)
# # seaborn heatmap 개념정리
# plt.show()
###########################################################


# ash열이 상관관계 제일 적은것을 확인.
xx= xx.drop(['ash','y~~'], axis=1)    # x데이터에서 ash열 제거
#print(xx)     #ash열이 제거되고 12개의 columns가 있는것 확인.

xx = xx.to_numpy()      #xx = xx.values       2가지방법중 아무거나 사용해서 xx를 다시 numpy로 바꿔준다. 
#왜냐하면 차원변환해줘야하는데 numpy만 된다하네
'''

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.9, shuffle=True, random_state=49)

scaler =MinMaxScaler()   #StandardScaler()RobustScaler()MaxAbsScaler()     

x_train = scaler.fit_transform(x_train).reshape(len(x_train),9,6,1)
x_test = scaler.transform(x_test).reshape(len(x_test),9,6,1)





#2.모델링

model = Sequential()
model.add(Conv2D(4,kernel_size=(2,1),strides=1,padding='valid', input_shape=(9,6,1), activation='relu')) #8,6,4
model.add(MaxPooling2D(2,2))                                                                             #4,3,4   
model.add(Conv2D(4,kernel_size=(2,1),strides=1,padding='valid', activation='relu'))                     # 3,3,4                                                                      # 1,1,10
model.add(Conv2D(4,kernel_size=(2,2),strides=1,padding='valid', activation='relu'))                     # 2,2,4
model.add(MaxPooling2D(2,2))                                                                            # 1 1 4                               
model.add(Flatten())       
model.add(Dense(40))
model.add(Dropout(0.5))
model.add(Dense(20))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras35_2_diabetes{krtime}.hdf5')
model.fit(x_train,y_train,epochs=10000, batch_size=1000,validation_split=0.111111, callbacks=[es])#,mcp



#4. 평가 예측

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

acc= str(round(loss[1], 4))

# a = inspect.getfile(inspect.currentframe())  #현재 파일이 위치한 경로 + 현재 파일 명
# print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
# print(a.split("\\")[-1]) #현재 파일 명

model.save(f"./_save/keras35_6_fetch_acc_Min_{acc}.h5")


'''
결과정리
            Minmax                  standard
loss:       0.7988 
accuracy:   0.6323

'''
