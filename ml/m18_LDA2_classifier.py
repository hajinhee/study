import numpy as np
from sklearn.datasets import fetch_covtype,load_iris, load_breast_cancer,load_wine

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터
# datasets = load_iris()
# datasets = load_breast_cancer()
datasets = load_wine()
# datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print('LDA 전:',x.shape)

x_train, x_test,y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state = 66, shuffle=True, stratify = y
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# stratify : 계층적인, 균등하게 빼준다.

# pca = PCA(n_components=28)
lda = LinearDiscriminantAnalysis()
# x = pca.fit_transform(x)
lda.fit(x_train,y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

# print(x)
print('LDA 후:',x_train.shape)  # (569, 1)

#2. 모델
from xgboost import XGBClassifier
model = XGBClassifier()

#3. 훈련
import time
start = time.time()
# model.fit(x_train, y_train, eval_metric='error')
model.fit(x_train, y_train, eval_metric='merror')
# model.fit(x_train, y_train)

end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print('결과:', results)
print('걸린 시간', end-start)

'''
- iris
LDA 전: (150, 4)
LDA 후: (120, 2)
결과: 1.0
걸린 시간 0.09819388389587402
- cancer
LDA 전: (569, 30)
LDA 후: (455, 1)
결과: 0.9473684210526315
걸린 시간 0.08024239540100098
- wine
LDA 전: (178, 13)
LDA 후: (142, 2)
결과: 1.0
걸린 시간 0.10049581527709961
-fetchcov
LDA 전: (581012, 54)
LDA 후: (464809, 6)
결과: 0.7878109859470065
걸린 시간 164.33383083343506
'''