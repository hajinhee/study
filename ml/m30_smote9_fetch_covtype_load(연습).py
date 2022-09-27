# 실습
# 증폭한 후 저장한 데이터를 불러와서
# 완성할것

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
import pickle

#1. 데이터
path='../_save/'

datasets = fetch_covtype()

x = datasets.data
y = datasets.target

# print(x.shape,y.shape)(581012, 54) (581012,)
# print(pd.Series(y).value_counts())
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747
# dtype: int64


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size = 0.8, stratify = y)

smote = SMOTE(random_state=66, k_neighbors=2)
x_train, y_train = smote.fit_resample(x_train, y_train)

path='./_save/'

x_train=np.load(path + 'm30_save_x_train_model.dat.npy')
y_train=np.load(path + 'm30_save_y_train_model.dat.npy')
# x_test=np.save(path + 'm30_save_x_test_model.dat', arr=x_test)
# y_test=np.save(path + 'm30_save_y_test_model.dat', arr=y_test)

# 스케일러 적용
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

import joblib
model=joblib.load(path + "m30_joblib3_save.dat")

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