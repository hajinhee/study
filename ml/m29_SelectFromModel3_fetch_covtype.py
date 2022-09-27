import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import fetch_covtype, load_boston
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(datasets.feature_names)   # numpy feature 확인 
print(x.shape, y.shape) # (581012, 54) (581012,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 66)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 2
kfold = KFold(n_splits = n_splits, shuffle = True, random_state=100)

parameters = [
    {'n_jobs': [-1],'n_estimators' : [10],'learning_rate' : [0.06],
    'max_depth' :[3]}]

#2. 모델 구성
model = GridSearchCV(XGBClassifier(eval_metric='mlogloss'), parameters, cv = kfold, verbose = 3,
                     refit=True, n_jobs=-1)

#3. 훈련 
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print('model.score :', score)
print(model.best_estimator_.feature_importances_)

aaa = np.sort(model.best_estimator_.feature_importances_) 
print(aaa)

for thresh in aaa:
    seletion = SelectFromModel(model.best_estimator_,threshold = thresh, prefit = True)
    select_x_train = seletion.transform(x_train)
    select_x_test = seletion.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)
    
    selection_model = XGBRegressor(n_jobs = -1)
    selection_model.fit(select_x_train, y_train, eval_metric = 'mlogloss')
    y_pred = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_pred)
    print('Thresh = %.3f, n=%d, R2: %.2f%%'
        %(thresh, select_x_train.shape[1], score*100))