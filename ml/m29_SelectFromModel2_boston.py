import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
# x, y = load_boston(return_X_y=True)
  # stratify = y >> yes가 아니고, y(target)이다.
print(datasets.feature_names)
'''
2, 4, 7,12 
['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT']
'''
x = np.delete(x, [2, 4, 7,12 ], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 66, )

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBRegressor(n_jobs = -1)

#3. 훈련 
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print('model.score :', score)

# print(model.feature_importances_)
# print(np.sort(model.feature_importances_))
# aaa = model.feature_importances_
# print(aaa)
# # # aaa = np.sort(model.feature_importances_)

'''
(404, 9) (102, 9)
Thresh = 0.014, n=9, R2: 93.06%

[0.01447933 0.00363372 0.01479118 0.00134153 0.06949984 0.30128664
 0.01220458 0.05182539 0.0175432  0.03041654 0.04246344 0.01203114
 0.4284835 ]
'''


# print('=====================================================')

# for thresh in aaa:
#     seletion = SelectFromModel(model,threshold = thresh, prefit = True)
#     select_x_train = seletion.transform(x_train)
#     select_x_test = seletion.transform(x_test)
#     print(select_x_train.shape, select_x_test.shape)
    
#     selection_model = XGBRegressor(n_jobs = -1)
#     selection_model.fit(select_x_train, y_train)
#     y_pred = selection_model.predict(select_x_test)
#     score = r2_score(y_test, y_pred)
#     print('Thresh = %.3f, n=%d, R2: %.2f%%'
#         %(thresh, select_x_train.shape[1], score*100))
