from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston, fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score
import time
# import warnings
# warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np

#1. 데이터
import pickle
path = './_save/'
datasets = pickle.load(open(path + 'm26_pickle1_save.datsets.dat','rb'))
x = datasets.data
y = datasets['target']

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
    # n_jobs = -1,  
    n_estimators = 10000,
    learning_rate = 0.025,
    # subsample_for_bin= 200000,
    max_depth = 4,
    min_child_weight = 1,
    subsample = 1,
    colsample_bytree = 1,
    reg_alpha = 0,              # 규제 : L1 = lasso
    # reg_lamda = 0,              # 규제 : L2 = ridge
    tree_method = 'gpu_hist',
    predictop = 'gpu_predictor',
    gpu_id=0,
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

'''
results : 0.7355
r2 : 0.7355
걸린 시간 : 79.7586

results : 0.7357
r2 : 0.7357
걸린 시간 : 4.8839

results : 0.9052
r2 : 0.9052
걸린 시간 : 353.217
'''