import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import time
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

# f1_score : 분류 값의 범위가 다를 때 사용한다.  >> 데이터가 불균형 할 때 

#1. 데이터
path = 'D:\\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', header = 0)  # sep=';' 컬럼 나눠주는 것, 미리 파일 열어서 확인하기
# index_col = None, : 인덱스 있는지 확인하기, header = 0,
## 데이터 확인
print(datasets.shape)   # (4898, 12)
print(datasets.head())
print(datasets.describe()) # 수치 데이터에 한해서 
print(datasets.info())

## numpy로 변경
datasets = datasets.values
print(type(datasets))     # <class 'numpy.ndarray'>

## numpy x, y 나누기

x = datasets[:, :11]  # 모든 행, 10번째 컬럼까지
y = datasets[:, 11]   # 모든 행, 11번째

print('라벨 : ',np.unique(y, return_counts = True))
# 라벨 :  (array([3., 4., 5., 6., 7., 8., 9.]), 
#      array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer


## 이상치 확인
# def outliers(data_out):
#     quartile_1, q2, quartile_3 = np.percentile(data_out,
#                                                [25, 50, 75])
#     print('1사분위 : ', quartile_1)
#     print('q2 :', q2)
#     print('3사분위 :', quartile_3)
#     iqr = quartile_3 - quartile_1
#     print('iqr :', iqr)
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     return np.where((data_out>upper_bound) | (data_out<lower_bound))

# outliers_loc = outliers(datasets)
# print('이상치의 위치 :', outliers(datasets))

## pandas로 하는법
# x = datasets.drop(["quality","alcohol","fixed acidity","citric acid"],axis=1)
# y = datasets["quality"]
# print(x.shape, y.shape)  # (4898, 11) (4898,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 66, stratify= y)  # stratify = y >> yes가 아니고, y(target)이다.

scaler = QuantileTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(
    n_jobs = -1,  
    n_estimators = 100,
    learning_rate = 0.054,
    # subsample_for_bin= 200000,
    max_depth = 3,
    min_child_weight = 1,
    subsample = 1,
    colsample_bytree = 1,
    reg_alpha = 1,              # 규제 : L1 = lasso
    # reg_lamda = 0,              # 규제 : L2 = ridge
    # tree_method = 'gpu_hist',
    # predictop = 'gpu_predictor',
    # gpu_id=0,
)


#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose = 3,
        #   eval_set=[(x_train,y_train),(x_test, y_test)],
          eval_metric='merror',              #rmse, mae, logloss, error
        #   early_stopping_rounds=2000,
          )
end = time.time()

result = model.score(x_test, y_test)
print('results :', round(result,4))

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('r2 :',round(acc,4))
print('걸린 시간 :', round(end-start, 4))
print('f1_score :', f1_score(y_test, y_pred, average= 'micro')) # >> average 해주어야 함

'''

'''


from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()