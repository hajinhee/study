# 1   357
# 0   212

# 라벨 0을 112개 삭제해서 재구성
# smote 넣어서 만들기 
# 넣은 것과 넣지 않은 것 비교

import pandas as pd 
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from imblearn.over_sampling import SMOTE
import time

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 66, stratify= y)  # stratify = y >> yes가 아니고, y(target)이다.
print(y)
print(pd.Series(y).value_counts())

# x_new = x[:-112]
# y_new = y[:-112]
print(np.where(0, y))


'''
1    357
0    212
'''