from sklearn.experimental import enable_halving_search_cv
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.svm import LinearSVC, SVC
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')
#1. 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 100)


parameters = [
    {'xg__max_depth' : [6,8,10],
    # 'xg__min_samples_leaf' :[3,5,7],
     'xg__eta' : [0.01, 0.2],
     'xg__colsample_bytree' : [0.5,0.6,0.7,1]}]

#2. 모델
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = make_pipeline(MinMaxScaler(),RandomForestClassifier())
pipe = Pipeline([('mm',MinMaxScaler()),('xg', XGBClassifier(eval_metric='merror'))])
# model = GridSearchCV(pipe, parameters, cv = 5, verbose = 3)
model = RandomizedSearchCV(pipe, parameters, cv = 5, verbose = 3)
# model = HalvingGridSearchCV(pipe, parameters, cv = 5, verbose = 3)

#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train, )
end = time.time()

#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('model.score :', result)
print('accuracy_score :', acc)
print('걸린 시간 :', end - start)
