from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston, fetch_covtype
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
datasets = fetch_covtype()
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

import pickle
path = './_save/'
pickle.dump(datasets, open(path + 'm26_pickle1_save.datsets.dat', 'wb'))

#2. 모델
model = XGBClassifier(
    n_jobs = -1,  
    n_estimators = 2000,
    learning_rate = 0.025,
    # subsample_for_bin= 200000,
    max_depth = 4,
    min_child_weight = 1,
    subsample = 1,
    colsample_bytree = 1,
    reg_alpha = 0,              # 규제 : L1 = lasso
    # reg_lamda = 0,              # 규제 : L2 = ridge
)

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose = 3,
          eval_set=[(x_train,y_train),(x_test, y_test)],
          eval_metric='mlogloss',              #rmse, mae, logloss, error
          early_stopping_rounds=2000,
          )
end = time.time()

result = model.score(x_test, y_test)
print('results :', round(result,4))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('r2 :',round(acc,4))
print('걸린 시간 :', round(end-start, 4))

print('=======================================')
# hist = model.evals_result()
# print(hist)

# 저장 
