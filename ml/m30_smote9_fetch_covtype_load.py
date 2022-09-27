from xgboost import XGBClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
import warnings
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y,
                 shuffle=True, random_state=66, train_size=0.8                                   
                                                    )

# import pickle
# path = "D:/_save_npy/"
# x_train = pickle.load(open(path + 'm30_smote_save_x_train.dat', 'rb'))
# y_train = pickle.load(open(path + 'm30_smote_save_y_train.dat', 'rb'))

# scaler = MinMaxScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
scaler = PolynomialFeatures()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = XGBClassifier(
                    # base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    # colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
                    # gamma=0, gpu_id=0, importance_type=None,
                    # interaction_constraints='', learning_rate=0.300000012,
                    # max_delta_step=0, max_depth=9, min_child_weight=1,
                    # monotone_constraints='()', n_estimators=5000, n_jobs=12,
                    # num_parallel_tree=1, objective='multi:softprob',
                    # predictor='gpu_predictor', random_state=0, reg_alpha=0,
                    # reg_lambda=1, scale_pos_weight=None, subsample=1,
                    # tree_method='gpu_hist', validate_parameters=1, verbosity=None
                      )


import joblib
model = joblib.load('D:\\Study\\_save\\m30_model.dat')

#3. 컴파일, 훈련
import time
start = time.time()
# model.fit(x_train, y_train, verbose=1,
#           eval_set=[(x_train, y_train), (x_test, y_test)],
#           eval_metric='mlogloss',                # rmse, mae, logloss, error
#           early_stopping_rounds=10,              # mlogloss, merror 
#           )     

end = time.time() - start


#4. 평가, 예측

score = model.score(x_test, y_test)
y_predict = model.predict(x_test)

print('라벨 : ', np.unique(y, return_counts=True))

print("걸린시간 : ", round(end, 4), "초")
print("model.score : ", score)
print("accuracy score : ", round(accuracy_score(y_test, y_predict),4))
print("f1 score : ", round(f1_score(y_test, y_predict, average='macro'),4))
