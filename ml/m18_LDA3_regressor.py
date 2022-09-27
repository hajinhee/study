import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_diabetes

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings(action='ignore')

#1. 데이터

# datasets = load_boston()
datasets = fetch_california_housing()
# datasets = load_diabetes()
# datasets = fetch_covtype()
x = datasets.data
y = datasets.target
y = np.round(y,0)
# y = y*10
print(y)
print('LDA 전:',x.shape)
b = []
for i in y:
    b.append(len(str(i).split('.')[1]))
print(np.unique(b,return_counts=True))

x_train, x_test,y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state = 66, shuffle=True
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# stratify : 계층적인, 균등하게 빼준다.

# pca = PCA(n_components=28)
lda = LinearDiscriminantAnalysis(n_components=2)
# x = pca.fit_transform(x)
lda.fit(x_train,y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

# print(x)
print('LDA 후:',x_train.shape)  # (569, 1)

#2. 모델
from xgboost import XGBClassifier, XGBRegressor
model = XGBRegressor()

#3. 훈련
import time
start = time.time()
# model.fit(x_train, y_train, eval_metric='error')
model.fit(x_train, y_train, eval_metric='rmse')
# model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print('결과:', results)
print('걸린 시간', end-start)

'''
- boston
LDA 전: (506, 13)
LDA 후: (404, 2)
결과: 0.73072844159542
걸린 시간 0.13970208168029785
- diabet
LDA 전: (442, 10)
LDA 후: (353, 2)
결과: 0.4274215749638751
걸린 시간 0.13663458824157715
- fetch_california_housing
LDA 전: (506, 13)
LDA 후: (404, 2)
결과: 0.73072844159542
걸린 시간 0.13970208168029785
-fetchcov
LDA 전: (581012, 54)
LDA 후: (464809, 6)
결과: 0.7878109859470065
걸린 시간 164.33383083343506
'''