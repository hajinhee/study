from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
# import warnings
# warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np

#1. 데이터
# datasets = fetch_california_housing()
datasets = load_boston()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)   # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle = True, random_state = 66, train_size=0.8
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 불러오기 // 모델, 훈련
# import pickle  
# import joblib
# 둘 다 가중치와 히스토리가 저장된다.
path = './_save/'
# model = pickle.load(open(path + 'm23_pickle1_save.dat','rb'))
# model = joblib.load(path + 'm24_joblib1_save.dat')
model = XGBRegressor()
model.load_model(path + 'm25_xgb1_save_model.dat')

#4. 평가
results = model.score(x_test, y_test)
print('result :', round(results,4))
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 :', round(r2,4))

print('=======================================')
hist = model.evals_result()
print(hist)