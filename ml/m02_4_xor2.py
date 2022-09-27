import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron   ## 원조
from sklearn.metrics import accuracy_score

###### poli #######

#1. 데이터   ### xor 데이터 ### 
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0, 1, 1, 0]

#2. 모델
# model = LinearSVC()
# model = Perceptron()     # xor에 막혔다. 
model = SVC()

#3. 훈련
model.fit(x_data, y_data)

#4. 평가, 예측
y_pred = model.predict(x_data)
print(x_data, " 의 예측결과 : ", y_pred)
results = model.score(x_data, y_data)
print('model.score : ', results)

acc = accuracy_score(y_data, y_pred)
print('accuracy_score : ', acc)


###polynomial###

'''
수학에서, 다항식(polynomial)은 한 개 이상의 항의 합으로 이루어진 식이다. 
즉, 단항식의 결합(덧셈과 뺄셈)으로 이루어진 식이다. 
예를 들어, x2 - 2x + 3, 4x3, 5xy + 6은 모두 다항식이다.
'''