from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler  
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.utils import to_categorical 
from sklearn.metrics import accuracy_score

from icecream import ic

#1.데이터 로드 및 정제
datasets = load_breast_cancer()
x = datasets.data  # (569, 30)
y = datasets.target  # (569,)
''''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0
 1 0 1 0 0 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1
...]
'''
y = to_categorical(y)  # 1차원 정수 배열을 (n, k) 크기의 2차원 배열로 변경(카테고리화) => (569, 2)
'''
[[1. 0.]
 [1. 0.]
 [1. 0.]
 ...
 [1. 0.]
 [1. 0.]
 [0. 1.]]
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=49)    

##################################### 스케일러 설정 옵션 ########################################
#scaler = MinMaxScaler()   
#scaler = StandardScaler()
#scaler = RobustScaler()
scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)    
x_test = scaler.transform(x_test) 



#2. 모델구성,모델링
input1 = Input(shape=(30))
dense1 = Dense(50)(input1)
dense2 = Dense(30)(dense1)
dense3 = Dense(15,activation='relu')(dense2)
dense4 = Dense(8,activation='relu')(dense3)
dense5 = Dense(5)(dense4)
output1 = Dense(2, activation='sigmoid')(dense5)  # 분류모델에서 출력 노드 수는 클래스 수와 동일해야 한다. 이진분류이므로 'sigmoid'
model = Model(inputs=input1,outputs=output1)

# model = Sequential()
# model.add(Dense(50, input_dim=30))
# model.add(Dense(30))
# model.add(Dense(15,activation="relu")) 
# model.add(Dense(8,activation="relu")) 
# model.add(Dense(5))
# model.add(Dense(2, activation='sigmoid'))
#model.summary()


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])  # 이진분류 'binary_crossentropy'
es = EarlyStopping(monitor='val_accuracy', patience=100, mode='max', verbose=1, baseline=None, restore_best_weights=True)
model.fit(x_train, y_train, epochs=10000, batch_size=10, validation_split=0.111111, callbacks=[es])

#4. 평가 
loss = model.evaluate(x_test, y_test)
print('loss: ', loss[0], 'accuracy: ', loss[1])

#5. 모델저장
model.save("./_save/keras25_3_save_cancer.h5")


'''
[loss]:  0.22260360419750214  [accuracy]:  0.9473684430122375
'''