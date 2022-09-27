import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from imblearn.over_sampling import SMOTE
from pprint import pprint
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (178, 13) (178,)
print(pd.Series(y).value_counts())

'''
1    71
0    59
2    48   >> data 30개를 추출한다. >> 슬라이싱을 사용한다.
===========================================================
1    71
0    59
2    18  >> 모두 71개로 맞춰준다.
'''

# 두 개 모두 슬라이싱 해준다.
x_new = x[:-30]
y_new = y[:-30]
print(pd.Series(y_new).value_counts())

pprint(y_new)
x_train, x_test, y_train, y_test = train_test_split(x_new,y_new,
                    train_size = 0.75, random_state = 66, shuffle = True,
stratify=y_new)
print(pd.Series(y_train).value_counts())
'''
1    53 >> 53개에 맞춰준다.
0    44
2    14
'''
model = XGBClassifier(n_jobs = -1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('model.score :', round(score,4))
y_pred = model.predict(x_test)
print('accuracy_score :', round(accuracy_score(y_test, y_pred),4))

pprint('=============================smote ===================================')

smote = SMOTE(random_state=66)
x_train, y_train = smote.fit_resample(x_train, y_train)
model = XGBClassifier(n_jobs = -1)
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('model.score :', round(score,4))
y_pred = model.predict(x_test)
print('accuracy_score :', round(accuracy_score(y_test, y_pred),4))
'''
model.score : 0.9778 > 그냥 실행
accuracy_score : 0.9778

model.score : 0.9459 > 데이터 축소
accuracy_score : 0.9459

odel.score : 0.973 > 데이터 증폭
accuracy_score : 0.973
'''