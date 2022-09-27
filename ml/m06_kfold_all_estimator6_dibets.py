from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)

# scaler = MinMaxScaler()         
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
n_splits = 5

kfold = KFold(n_splits = n_splits, shuffle = True, random_state=100)

model = XGBRegressor()
scores = cross_val_score(model, x_train, y_train, cv = kfold)
print('ACC :',scores, "\ncross_val_score :", round(np.mean(scores),4))


'''
ACC : [0.24318051 0.23162622 0.26718244 0.32721805 0.59239959] 
cross_val_score : 0.3323
'''