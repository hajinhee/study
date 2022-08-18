from tensorflow.keras.models import Sequential          
from tensorflow.keras.layers import Dense
import numpy as np 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

#1. 데이터 
datasets = load_wine()
print(datasets.DESCR)  # Instances: 150개, Attributes: 13개, class: 3개(class_0 = 0 , class _1 = 1, class_2 =2)
print(datasets.feature_names) 
'''
['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 
'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
'''                                                      
x = datasets.data 
y = datasets.target

print(x.shape, y.shape)  # (178,13) (178,)
print(np.unique(y)) # [0 1 2]

# one hot encoding
y = to_categorical(y)
print(y.shape)  # (178, 3)으로 바뀌어져 있음
print(np.unique(y)) # [1,0,0],[0,1,0],[0,0,1]로 바뀌어져있다.

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)


#2. 모델링 모델구성
model = Sequential()
model.add(Dense(120, activation='linear', input_dim=13))    
model.add(Dense(100, activation='linear'))   
model.add(Dense(80, activation='linear'))
model.add(Dense(60, activation='linear'))
model.add(Dense(40, activation='linear'))
model.add(Dense(20, activation='linear'))
model.add(Dense(3, activation='softmax'))   


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
es = EarlyStopping(monitor = "val_loss", patience=50, mode='min',verbose=1,restore_best_weights=True)
model.fit(x_train,y_train,epochs=10000, batch_size=1, verbose=1,validation_split=0.2, callbacks=[es])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   
print('loss : ', loss[0])  # loss :  0.1840820461511612
print('accuracy : ', loss[1])  # accuracy :  0.9166666865348816

results = model.predict(x_test[:7])
print(x_test[:7])
print(y_test[:7])
print(results)