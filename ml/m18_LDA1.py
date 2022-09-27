from sklearn.decomposition import PCA
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import fetch_covtype

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape)   # (569, 30)

# pca = PCA(n_components=28)
lda = LinearDiscriminantAnalysis()
# x = pca.fit_transform(x)
lda.fit(x,y)
x = lda.transform(x)

# print(x)
print(x.shape)  # (569, 1)
x_train, x_test,y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state = 66, shuffle=True
)

#2. 모델
from xgboost import XGBRegressor, XGBClassifier
model = XGBClassifier()

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print('결과:', results)
print('걸린 시간', end-start)

'''
- xgboost
결과: 0.9736842105263158
- pca 

- LDA (지도학습)
결과: 0.9824561403508771

ValueError: n_components cannot be larger than min(n_features, n_classes - 1).
                                                  (컬럼, y)
'''

'''
fetch_covtype
결과: 0.7882498730669604

'''