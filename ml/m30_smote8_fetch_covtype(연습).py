#실습
#증폭한 후 저장 f1
import lightgbm
import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMClassifier

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(np.unique(y, return_counts=True))  #  (array([0, 1]), array([212, 357], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8, stratify = y)

print(np.unique(y_train, return_counts=True)) # [ 146, 1166, 1758,  704,  144]

# Smote 적용(데이터 증폭하기)
smote = SMOTE(random_state=66, k_neighbors=2)
x_train, y_train = smote.fit_resample(x_train, y_train)

import pickle
path='./_save/'

np.save(path + 'm30_save_x_train_model.dat', arr=x_train)
np.save(path + 'm30_save_y_train_model.dat', arr=y_train)
np.save(path + 'm30_save_x_test_model.dat', arr=x_test)
np.save(path + 'm30_save_y_test_model.dat', arr=y_test)

print(np.unique(y_train, return_counts=True))  

# 스케일러 적용
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9,
              enable_categorical=False, eval_metric='merror', gamma=0,
              gpu_id=-1, importance_type=None, interaction_constraints='',
              learning_rate=0.3, max_delta_step=0, max_depth=6,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=300, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', predictor= 'gpu_predictor', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None 
)

#3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time()-start
print("걸린 시간 : ", round(end, 2))

# path='../_save/'
# model.save_model(path + 'm30_lgbm1_save_model.dat')
# pickle.dump(smote, open(path + 'm30_smote_pickle_save.dat', 'wb'))

#4. 평가, 예측
pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test,pred,average='macro')

print('acc : ', acc)
print('f1 : ', f1)

path='./_save/'
import joblib
joblib.dump(model, path + "m30_joblib3_save.dat")

