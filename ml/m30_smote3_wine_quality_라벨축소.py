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
print(pd.Series(y).value_counts())

'''
newlist = []

for i in y:
      # print(i)
      if i <= 4:
            newlist += [0]
      elif  i <= 7:
            newlist += [1]
      else:
            newlist += [2]
'''
# y = np.array(newlist)
# print(np.unique(y, return_counts=True))

# x_train, x_test, y_train, y_test = train_test_split(x,y,
#         train_size =0.8, shuffle=True, random_state = 66, stratify= y)  # stratify = y >> yes가 아니고, y(target)이다.

# smote = SMOTE(random_state=66, k_neighbors=2)
# x_train, y_train = smote.fit_resample(x_train, y_train)
# model = XGBClassifier(n_jobs = -1)
# start = time.time()
# model.fit(x_train, y_train)
# end = time.time()
# score = model.score(x_test, y_test)
# print('model.score :', round(score,4))
# y_pred = model.predict(x_test)
# print('accuracy_score :', round(accuracy_score(y_test, y_pred),4))
# f1 = f1_score(y_test, y_pred, average='macro')
# print('f1_score:',f1)
# print('걸린시간 : ', end-start)

# '''
# f1_score: 0.4096885097111477
# 걸린시간 :  0.96919846534729
# --------------------------------
# f1_score: 0.6366979600479988
# 걸린시간 :  1.2325246334075928
# '''