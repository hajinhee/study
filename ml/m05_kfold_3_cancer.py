from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import numpy as np


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=100)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
n_splits = 5

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
kfold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state=100)

model = SVC()

scores = cross_val_score(model, x_train, y_train, cv = kfold)
print('ACC :',scores, "\ncross_val_score :", round(np.mean(scores),4))

'''
- kfold
ACC : [0.97802198 0.94505495 0.98901099 0.98901099 0.98901099] 
cross_val_score : 0.978
- stratified kfold
ACC : [0.96703297 0.98901099 0.98901099 0.95604396 1.        ] 
cross_val_score : 0.9802
'''