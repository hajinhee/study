import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.datasets import fetch_covtype
from imblearn.over_sampling import SMOTE
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import time
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential, load_model

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) 
print(pd.Series(y).value_counts())
'''
2    283301
1    211840
3     35754
7     20510
6     17367
5      9493
4      2747
'''
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 66, stratify= y)  # stratify = y >> yes가 아니고, y(target)이다.

# smote = SMOTE(random_state=66, k_neighbors=1)
# x_train, y_train = smote.fit_resample(x_train, y_train)

# scaler = QuantileTransformer()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# np.save('./_save_npy/m30_x_train.npy', arr = x_train)
# np.save('./_save_npy/m31_x_train.npy', arr = x_train)


# #2. 모델
# model = XGBClassifier(
#     # # n_jobs = -1,  
#     # n_estimators = 1000,
#     # learning_rate = 0.025,
#     # # subsample_for_bin= 200000,
#     # max_depth = 4,
#     # min_child_weight = 1,
#     # subsample = 1,
#     # colsample_bytree = 1,
#     # reg_alpha = 0,              # 규제 : L1 = lasso
#     # reg_lamda = 0,              # 규제 : L2 = ridge
#     # tree_method = 'gpu_hist',
#     # predictop = 'gpu_predictor',
#     # gpu_id=0,
# )

model = load_model('./_save/m30_model.dat')
start = time.time()
model.fit(x_train, x_test)
end = time.time()

score = model.score(x_test, y_test)
print('model.score :', round(score,4))

y_pred = model.predict(x_test)
print('accuracy_score :', round(accuracy_score(y_test, y_pred),4))
f1 = f1_score(y_test, y_pred, average='macro')
print('f1_score:',f1)
print('걸린시간 : ', end-start)

# # 저장 
# import pickle
# path = './_save/'
# pickle.dump(model, open(path + 'm30_smote_save.dat', 'wb'))