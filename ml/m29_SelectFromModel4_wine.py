import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.datasets import load_diabetes, load_boston
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')


#1. 데이터
path = 'D:\\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', header = 0)
datasets = datasets.values

x = datasets[:,:11]  
y = datasets[:, 11]

#print(type(x)) # numpy
x = np.delete(x,[0,6],axis=1)

x_train, x_test, y_train, y_test = train_test_split (x, y, shuffle=True, random_state=66, train_size=0.8)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [10, 12], 'min_samples_leaf' :[3, 7, 10], 'min_samples_split' : [3, 5]},
    {'n_estimators' : [100], 'max_depth' : [6, 12], 'min_samples_leaf' :[7, 10], 'min_samples_split' : [2, 3]}] # 'eval_metric':['merror']]

model = GridSearchCV(XGBClassifier(), parameters, cv=kfold, verbose=1, refit=True)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print('model.score: ', score)
''' 
model.score:  0.7091836734693877
'''
#print(model.best_estimator_.feature_importances_)
''' 
[0.07118042 0.11068212 0.07296681 0.08077707 0.07586595 0.08884557
 0.06964406 0.07237581 0.07518705 0.07214403 0.21033108]     
'''
#print(np.sort(model.best_estimator_.feature_importances_))  # 오름차순으로 정렬해주기
''' 
[0.06964406 0.07118042 0.07214403 0.07237581 0.07296681 0.07518705
 0.07586595 0.08077707 0.08884557 0.11068212 0.21033108]
'''
aaa = np.sort(model.best_estimator_.feature_importances_)  # 오름차순으로 정렬해주기

# print("==============================================================================")
# for thresh in aaa:
#     selection = SelectFromModel(model.best_estimator_, threshold = thresh, prefit = True)   
    
#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     print(select_x_train.shape, select_x_test.shape)
    
#     selection_model = XGBRegressor(n_jobs = -1)
#     selection_model.fit(select_x_train, y_train, eval_metric='merror')
    
#     y_predict = selection_model.predict(select_x_test)
#     score = r2_score(y_test, y_predict)
#     print("Thresh = %.3f, n=%d, R2: %2f%%"
#           %(thresh, select_x_train.shape[1], score*100))
  
##################################################################################
''' 
(3918, 11) (980, 11)
Thresh = 0.070, n=11, R2: 49.008384%
(3918, 10) (980, 10)
Thresh = 0.071, n=10, R2: 47.242234%
(3918, 9) (980, 9)
Thresh = 0.072, n=9, R2: 49.303631%  # 성능 제일 good
(3918, 8) (980, 8)
Thresh = 0.072, n=8, R2: 48.556362%
(3918, 7) (980, 7)
Thresh = 0.073, n=7, R2: 47.938542%
(3918, 6) (980, 6)
Thresh = 0.075, n=6, R2: 45.368998%
(3918, 5) (980, 5)
...............
'''

""" 
기존 model.score)  0.7091836734693877
컬럼 제거 후 model.score) 0.7030612244897959
"""