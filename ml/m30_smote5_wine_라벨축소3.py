'''
실습
3,4 > 0
5,6,7 > 1
8,9 >2

3,4,5 > 0
6 > 1
7,8,9 > 2

'''

import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from imblearn.over_sampling import SMOTE
import time
#1. 데이터
path = 'D:\\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', header = 0)
datasets = datasets.values

x = datasets[:, :11]
y = datasets[:, 11]

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 66, stratify= y)  # stratify = y >> yes가 아니고, y(target)이다.

model = XGBClassifier(n_jobs = -1)
start = time.time()
model.fit(x_train, y_train)
end = time.time()
score = model.score(x_test, y_test)
print('model.score :', round(score,4))
y_pred = model.predict(x_test)
print('accuracy_score :', round(accuracy_score(y_test, y_pred),4))
f1 = f1_score(y_test, y_pred, average='macro')
print('f1_score:',f1)
print('걸린시간 : ', end-start)

print('=========================================================================')


# print('라벨 : ',np.unique(y, return_counts = True))
# print(pd.Series(y).value_counts())
'''
6.0    2198
5.0    1457
7.0     880
8.0     175
4.0     163
3.0      20
9.0       5
'''
#### 선생님 방법

for index, value in enumerate(y):
    if value == 9:
        y[index] = 2    
    elif value == 8:
        y[index] = 2
    elif value == 7:
        y[index] = 2
    elif value == 6:
        y[index] = 1
    elif value == 5:
        y[index] = 0
    elif value == 4:
        y[index] = 0
    elif value == 3:
        y[index] = 0
    else:
        y[index] = 0

print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 66, stratify= y)  # stratify = y >> yes가 아니고, y(target)이다.

# smote = SMOTE(random_state=66, k_neighbors=2)
# x_train, y_train = smote.fit_resample(x_train, y_train)
model = XGBClassifier(n_jobs = -1)
start = time.time()
model.fit(x_train, y_train)
end = time.time()
score = model.score(x_test, y_test)
print('model.score :', round(score,4))
y_pred = model.predict(x_test)
print('accuracy_score :', round(accuracy_score(y_test, y_pred),4))
f1 = f1_score(y_test, y_pred, average='macro')
print('f1_score:',f1)
print('걸린시간 : ', end-start)


'''
smote 전
accuracy_score : 0.9367
f1_score: 0.5922685681299195
걸린시간 :  0.4637594223022461

smote 후
accuracy_score : 0.9286
f1_score: 0.6366979600479988
걸린시간 :  1.340501070022583

smote 전
accuracy_score : 0.7173
f1_score: 0.7145586085311318
걸린시간 :  0.5265843868255615

smote 후
accuracy_score : 0.7224
f1_score: 0.7205080504063961
걸린시간 :  0.683100700378418
'''