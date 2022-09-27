import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import time
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from imblearn.over_sampling import SMOTE
#1. 데이터
path = 'D:\\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', header = 0)
datasets = datasets.values
x = datasets[:, :11]  # 모든 행, 10번째 컬럼까지
y = datasets[:, 11]
print('라벨 : ',np.unique(y, return_counts = True))

x_train, x_test, y_train, y_test = train_test_split(x,y,
                    train_size = 0.75, random_state = 66, shuffle = True,
                stratify=y)

model = XGBClassifier(n_jobs = -1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('model.score :', round(score,4))
y_pred = model.predict(x_test)
print('accuracy_score :', round(accuracy_score(y_test, y_pred),4))

print(pd.Series(y_train).value_counts())
f2 = f1_score(y_test, y_pred, average='macro')
print('f1_score:',f2)
print('=============================smote ===================================')

smote = SMOTE(random_state=66, k_neighbors=2)
x_train, y_train = smote.fit_resample(x_train, y_train)
model = XGBClassifier(n_jobs = -1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('model.score :', round(score,4))
y_pred = model.predict(x_test)
print('accuracy_score :', round(accuracy_score(y_test, y_pred),4))
f1 = f1_score(y_test, y_pred, average='macro')
print('f1_score:',f1)

'''
6.0    1648
5.0    1093
7.0     660
8.0     131
4.0     122
3.0      15
9.0       4
'''
'''
- 기본
model.score : 0.6433
accuracy_score : 0.6433

model.score : 0.6318
accuracy_score : 0.6318

SMOTE는 k-neighbors 비슷한 위치에 데이터를 추가하여 증폭시킨다. 
최근접 이웃
'''

