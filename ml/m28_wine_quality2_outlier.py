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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer

# f1_score : 분류 값의 범위가 다를 때 사용한다.  >> 데이터가 불균형 할 때 

#1. 데이터
path = 'D:\\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', header = 0)  # sep=';' 컬럼 나눠주는 것, 미리 파일 열어서 확인하기
print(datasets.shape)
############## 아웃라이어 확인 ####################
# from sklearn.covariance import EllipticEnvelope
# outliers = EllipticEnvelope(contamination=.15)
# pred = outliers.fit_predict(datasets)
# print(pred)

def boxplot_vis(data, target_name):
    plt.figure(figsize=(30, 30))
    for col_idx in range(len(data.columns)):
        # 6행 2열 서브플롯에 각 feature 박스플롯 시각화
        plt.subplot(6, 2, col_idx+1)
        # flierprops: 빨간색 다이아몬드 모양으로 아웃라이어 시각화
        plt.boxplot(data[data.columns[col_idx]], flierprops = dict(markerfacecolor = 'r', marker = 'D'))
        # 그래프 타이틀: feature name
        plt.title("Feature" + "(" + target_name + "):" + data.columns[col_idx], fontsize = 20)
    # plt.savefig('../figure/boxplot_' + target_name + '.png')
    plt.show()
boxplot_vis(datasets,'white_wine')

def remove_outlier(input_data):
    q1 = input_data.quantile(0.25) # 제 1사분위수
    q3 = input_data.quantile(0.75) # 제 3사분위수
    iqr = q3 - q1 # IQR(Interquartile range) 계산
    minimum = q1 - (iqr * 1.5) # IQR 최솟값
    maximum = q3 + (iqr * 1.5) # IQR 최댓값
    # IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치)
    df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
    return df_removed_outlier

prep = remove_outlier(datasets)

# 목표변수 할당
prep['target'] = 0
a = prep.isnull().sum()
print(a)

'''
fixed acidity           146
volatile acidity        186
citric acid             270
residual sugar            7
chlorides               208
free sulfur dioxide      50
total sulfur dioxide     19
density                   5
pH                       75
sulphates               124
alcohol                   0
quality                 200
target                    0
'''
prep.dropna(axis = 0, how = 'any', inplace = True)
print(f"이상치 포함된 데이터 비율: {round((len(datasets) - len(prep))*100/len(datasets), 2)}%")

print(prep.shape)

x = prep.drop("quality",axis=1)
y = prep["quality"]

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 66, stratify= y)  # stratify = y >> yes가 아니고, y(target)이다.

scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(
    n_jobs = -1,  
    n_estimators = 100,
    learning_rate = 0.054,
    # subsample_for_bin= 200000,
    max_depth = 6,
    min_child_weight = 1,
    subsample = 1,
    colsample_bytree = 1,
    reg_alpha = 1,              # 규제 : L1 = lasso
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
print('f1_score :', f1_score(y_test, y_pred, average= 'micro')) 

from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()