from sklearn.decomposition import PCA
import numpy as np
from sklearn.datasets import load_boston, fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape)   # (506, 13)

pca = PCA(n_components=30)
x = pca.fit_transform(x)
# print(x)
print(x.shape)  # (506, 8)
x_train, x_test,y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state = 66, shuffle=True
)
pca_EVR = pca.explained_variance_ratio_
print(sum(pca_EVR))
print(pca_EVR)
cumsum = np.cumsum(pca_EVR)
print(cumsum)       #누적합

import matplotlib.pyplot as plt
plt.plot(cumsum)
# plt.plot(pca_EVR)
plt.grid()
plt.show()
# #2. 모델
# from xgboost import XGBRegressor, XGBClassifier
# model = XGBClassifier()

# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# results = model.score(x_test, y_test)
# print('결과:', results)

'''
(569, 30)
(569, 10)
[9.82044672e-01 1.61764899e-02 1.55751075e-03 1.20931964e-04
 8.82724536e-05 6.64883951e-06 4.01713682e-06 8.22017197e-07
 3.44135279e-07 1.86018721e-07]
'''