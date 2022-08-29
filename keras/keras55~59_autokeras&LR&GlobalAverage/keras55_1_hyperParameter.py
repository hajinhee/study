from tabnanny import verbose
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, Input
from tensorflow.keras.utils import to_categorical
from icecream import ic
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor 
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import os

#1. load data
(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28*28).astype('float32')/255  # int -> float, Minmax, 3d -> 2d
x_test = x_test.reshape(10000, 28*28).astype('float32')/255         
ic(np.unique(y_train, return_counts=True))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> 다중분류

# one hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
ic(len(y_train))  # 60000

#2. modeling
def build_model(drop=0.5, optimizer='adam', activation='relu'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation=activation, name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation=activation, name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation=activation, name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs,outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs = [32, 64, 128, 256, 512]
    optimizers = ['adam', 'rmsprop', 'adadelta']
    dropout = [0.3, 0.4, 0.5]
    activation = ['relu', 'linear', 'sigmoid', 'selu', 'elu']
    return {'batch_size' : batchs, 'optimizer' : optimizers, 'drop' : dropout, 'activation' : activation}

hyperparameters = create_hyperparameter()
keras_model = KerasClassifier(build_fn=build_model, verbose=1)  
model = GridSearchCV(keras_model, param_grid=hyperparameters, cv=3, verbose=1)
<<<<<<< HEAD

'''
GridSearchCV
-estimator : classifier, regressor, pipeline 등 가능
-param_grid : 튜닝을 위해 파라미터, 사용될 파라미터를 dictionary 형태로 만들어서 넣는다.
-cv : 교차 검증에서 몇개로 분할되는지 지정한다.
-scoring : 예측 성능을 측정할 평가 방법을 넣는다. 보통 'accuracy' 로 지정하여서 정확도로 성능 평가를 한다.
-refit : True가 디폴트로 True로 하면 최적의 하이퍼 파라미터를 찾아서 재학습 시킨다.
'''

=======
'''
GridSearchCV
-estimator : classifier, regressor, pipeline 등 가능
-param_grid : 튜닝을 위해 파라미터, 사용될 파라미터를 dictionary 형태로 만들어서 넣는다.
-cv : 교차 검증에서 몇개로 분할되는지 지정한다.
-scoring : 예측 성능을 측정할 평가 방법을 넣는다. 보통 'accuracy' 로 지정하여서 정확도로 성능 평가를 한다.
-refit : True가 디폴트로 True로 하면 최적의 하이퍼 파라미터를 찾아서 재학습 시킨다.
'''

>>>>>>> 3c53c44b23e9b788a428b83c1cc067a4275d80a8
#3. train
start = time.time()
model.fit(x_train, y_train, verbose=1, epochs=30, validation_split=0.2)
end = time.time()

print('걸린 시간: ', end-start)
print('model.best_params_ : ', model.best_params_)  # 최고 점수를 낸 파라미터
print('model.best_estimator_', model.best_estimator_)  # 최고 점수를 낸 파라미터를 가진 모형
print('model.best_score_', model.best_score_)  # 최고 점수
print('model.score', model.score)

#4. predict
y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_test, y_predict))

# 가중치 save
model.save('./_save/keras55_1_save_model.h5')
model.save_weights('./_save/keras55_1_save_weights.h5')