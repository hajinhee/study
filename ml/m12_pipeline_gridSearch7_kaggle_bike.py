from sklearn.experimental import enable_halving_search_cv
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.svm import LinearSVC, SVC
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

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
        train_size =0.8, shuffle=True, random_state = 100)


parameters = [
    {'xg__max_depth' : [6,8,10],
    # 'xg__min_samples_leaf' :[3,5,7],
     'xg__eta' : [0.01,0.05,0.1,0.15, 0.2],
     'xg__colsample_bytree' : [0.5,0.6,0.7,1]}]

#2. 모델
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = make_pipeline(MinMaxScaler(),RandomForestClassifier())
pipe = Pipeline([('mm',MinMaxScaler()),('xg', XGBRegressor(eval_metric='merror'))])
# model = GridSearchCV(pipe, parameters, cv = 5, verbose = 3)
# model = RandomizedSearchCV(pipe, parameters, cv = 5, verbose = 3)
model = HalvingGridSearchCV(pipe, parameters, cv = 5, verbose = 3)

#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train)
end = time.time()

#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print('model.score :', result)
print('accuracy_score :', r2)
print('걸린 시간 :', end - start)

'''
- RandomizedSearchCV
model.score : 0.30905505736932193
accuracy_score : 0.30905505736932193
걸린 시간 : 18.051002740859985
- GridSearchCV
model.score : 0.30905505736932193
accuracy_score : 0.30905505736932193
걸린 시간 : 50.939159870147705
- HalvingGridSearchCV
model.score : 0.3342082325728035
accuracy_score : 0.3342082325728035
걸린 시간 : 89.0862295627594
'''