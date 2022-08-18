from tensorflow.keras.models import Sequential          
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical # 원핫인코딩 도와주는 함수 기능   
from sklearn.datasets import load_iris  # 꽃잎의 모양과 줄기 넓이 등의 특징으로 어떤 꽃인지 판별하는 데이터셋

# 1. 데이터 정제 <-- 이 단계에서 원핫인코딩 
datasets = load_iris()
print(datasets.DESCR)  # Instances: 150개, Attributes: 4개, class(꽃의 종류): 3개
'''
Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica
'''
print(datasets.feature_names) # 컬럼명
'''
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
'''

x = datasets.data  # (150, 4)
y = datasets.target  # (150,)

print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''
print(np.unique(y))  # [0 1 2]

# one hot encoding
y = to_categorical(y)
print(y)        # [1,0,0],[0,1,0],[0,0,1]
print(y.shape)  # (150, 3)으로 바뀌어있다.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape, y_train.shape) # (120, 4) (120, 3)
print(x_test.shape, y_test.shape)   # (30, 4) (30, 3)


#2. 모델링 모델구성
model = Sequential()
model.add(Dense(120, activation='linear', input_dim=4))    
model.add(Dense(100, activation='linear'))   
model.add(Dense(80, activation='linear'))
model.add(Dense(60, activation='linear'))
model.add(Dense(40, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(3, activation='softmax'))   


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 다중분류는 'categorical_crossentropy'

es = EarlyStopping(monitor = "val_loss", patience=50, mode='min', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=300, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   
print('loss : ' , loss[0]) # 0.06474184989929199
print('accuracy : ', loss[1]) # 0.9666666388511658

results = model.predict(x_test[:7])
print(y_test[:7])
'''
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]]
'''
print(results)
'''
[[6.53624593e-04 9.98846531e-01 4.99865389e-04]
 [1.13535985e-04 9.87196088e-01 1.26903988e-02]
 [7.21414253e-05 9.70304430e-01 2.96233334e-02]
 [9.99984980e-01 1.50336346e-05 5.26253882e-24]
 [4.40584234e-04 9.98531938e-01 1.02750387e-03]
 [9.38222918e-04 9.98795986e-01 2.65790877e-04]
 [9.99979377e-01 2.06511140e-05 1.23959031e-23]]
'''