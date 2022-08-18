from tabnanny import verbose
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Input

#1. 데이터

(x_train, y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255          # 정수형 -> 실수형으로 바꿔줌 + Minmax + 2차원데이터로 바꿔서 Dense하겠다

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)                                     # 오랜만에 해주는 원핫인코딩. 이게 싫으면 loss에서 sparse_categorical_crossentropy 해주면된다.

#2. 모델

def build_model(drop=0.5, optimizer='adam', activation='relu'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10,activation='softmax',name='outputs')(x)
    
    model = Model(inputs=inputs,outputs=outputs)
    
    model.compile(optimizer=optimizer, metrics=['acc'],
                  loss='categorical_crossentropy')
    
    return model

def create_hyperparameter():
    batchs = [100,200,300,400,500]
    optimizers = ['adam', 'rmsprop', 'adadelta']                # 그라디언트 폭발과 소멸 optimizers를 진짜 잘 조정해야 튜닝의 고수다.
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu','linear','sigmoid','selu','elu']
    return {"batch_size" : batchs, "optimizer" : optimizers, "drop" : dropout,
            "activation" : activation}

hyperparameters = create_hyperparameter()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor   # KerasClassifier,KerasRegressor 2가지형태가있다.
keras_model = KerasClassifier(build_fn=build_model, verbose=1)  # tensorflow를 scikit_learn형태로 감싸준다.  

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
model = GridSearchCV(keras_model, hyperparameters, cv=3, verbose=1)

import time
start = time.time()
model.fit(x_train,y_train, verbose=1, epochs=30, validation_split=0.2)
#estimator should be an estimator implementing 'fit' method, <function build_model at 0x000001FD710C9160> was passed
#GridSearchCV는 머신러닝에서 썼었고 지금은 딥러닝이다. 서로 짝을 맞춰줘야한다.
end = time.time()

print("걸린 시간은요 ~ : ", end - start)
print("model.best_params_ : ",model.best_params_)
print("model.best_estimator_",model.best_estimator_)
print("model.best_score_",model.best_score_)
print("model.score",model.score)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test,y_predict))

# 가중치 save
import os
path = os.getcwd() + '/'
model.save(path + "keras55_1_save_model.h5")
model.save_weights(path + "keras55_1_save_weights.h5")