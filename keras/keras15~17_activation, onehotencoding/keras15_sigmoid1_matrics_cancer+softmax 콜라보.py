from tensorflow.keras.models import Sequential          
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical # 원핫 인코딩 도와주는 함수 기능 
 
#1. 데이터 정제
datasets = load_breast_cancer()
x = datasets.data  # (569, 30)
y = datasets.target  # (569, 2)
# y = to_categorical(y)
# print(datasets.DESCR) 
# print(datasets.feature_names)
'''
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
'''
# print(y)  # 결과값이 0,1 밖에 없는걸 보는 순간 2진분류인거 판단 & loss는 binary cross Entropy랑 sigmoid() 함수인거 까지 자동으로 생각
'''
[[1. 0.]
 [1. 0.]
 [1. 0.]
 ...
 [1. 0.]
 [1. 0.]
 [0. 1.]]
'''
# print(np.unique(y))  # 분류값에서 unique한 것이 무엇인지 볼 수 있음
'''
[0. 1.]
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42) 

#2. 모델링 
model = Sequential()
model.add(Dense(30, activation='linear', input_dim=30))  # activation='linear' -> 원래 값을 그대로 넘겨준다(default로 생략 가능)
model.add(Dense(25, activation='linear'))  # 값이 너무 큰거같으면 중간에 sigmoid한번써서 줄여줄 수도 있다(레이어를 거치며 커져버린 y=wx+b의 값들을 0.0~1.0사이로 잡아주는 역할)
model.add(Dense(20, activation='linear'))
model.add(Dense(15, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(5, activation='linear'))
model.add(Dense(1, activation='sigmoid'))  # 이진분류 sigmoid, 다중분류 softmax 

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # sigmoid = binary_crossentropy(이진분류), softmax = categorical_crossentropy(다중분류)   
# matrics=['accuracy']는 그냥 r2 score처럼 지표를 보여주는 것으로 fit에 영향을 끼치지않는다
# accuracy는 평가 지표일 뿐, loss가 중요하다(설령 accuracy가 높다고 해도 loss값이 크면 모델이 좋다고 할 수 없다)
# loss와 val_loss를 따지면 val_loss가 더 믿을만하다.

es = EarlyStopping(monitor = "val_loss", patience=50, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=300, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가
loss = model.evaluate(x_test, y_test) # 평가 결과 = loss, accuracy 값
print('loss, accuracy : ' , loss) # [0.2333454042673111, 0.9035087823867798]
# evaluate의 결과 값은 원래 loss 1개지만, matrics=['accuracy'] 를 사용했기 때문에 loss와 accuracy가 함께 출력된다.
# evaluate의 첫 번째 값은 무조건 loss, 그 후에는 사전에 설정해준 값들이 순차적으로 나온다.

#5. 예측
y_predict = model.predict(x_test)
print(y_predict)
print(y_test)

