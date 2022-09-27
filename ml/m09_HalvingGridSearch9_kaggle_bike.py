from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score, mean_squared_error #mse
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import time

#1. 데이터 
path = '../_data/kaggle/bike/'  
train = pd.read_csv(path+'train.csv')  
# print(train)      # (10886, 12)
test_file = pd.read_csv(path+'test.csv')
# print(test.shape)    # (6493, 9)
submit_file = pd.read_csv(path+ 'sampleSubmission.csv')
x = train.drop(['datetime', 'casual','registered','count'], axis=1) # axis=1 컬럼 삭제할 때 필요함
test_file = test_file.drop(['datetime'], axis=1) 

y = train['count']
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.9, shuffle=True, random_state = 42)

n_splits = 3
kfold = KFold(n_splits = n_splits, shuffle = True, random_state=100)

parameters = [
    {'n_estimators': [100,200,300],'max_depth' : [6,8,10,12]},
    {'min_samples_leaf' :[3,5,7,10],'min_samples_split' : [2,3,5,10]}]
                                                         # 총 112번(더해준다) 


#2. 모델 구성
model = HalvingGridSearchCV(RandomForestRegressor(), parameters, cv = kfold, verbose = 3,
                     refit=True, n_jobs=1)

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
print('최적의 매개 변수 :', model.best_estimator_)
print('최적의 파라미터 : ', model.best_params_)
print('best_score_ :', model.best_score_)
print('소요 시간 :', end-start)

'''
최적의 매개 변수 : RandomForestRegressor(min_samples_leaf=10, min_samples_split=3)
최적의 파라미터 :  {'min_samples_leaf': 10, 'min_samples_split': 3}
best_score_ : 0.3471854102674424
소요 시간 : 25.13172459602356

최적의 매개 변수 : XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
             gamma=0, gpu_id=-1, importance_type=None,
             interaction_constraints='', learning_rate=0.300000012,
             max_delta_step=0, max_depth=8, min_child_weight=1, missing=nan,
             monotone_constraints='()', n_estimators=100, n_jobs=12,
             num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
             reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
             validate_parameters=1, verbosity=None)
최적의 파라미터 :  {'max_depth': 8, 'n_estimators': 100}
best_score_ : 0.22255552044907165
소요 시간 : 36.30458045005798
'''