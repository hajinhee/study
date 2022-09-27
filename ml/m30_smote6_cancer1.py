# smote 넣어서 만들기 
# 넣은 것과 넣지 않은 것 비교
import pandas as pd 
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from imblearn.over_sampling import SMOTE
import time

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)  # (569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 66, stratify= y)  # stratify = y >> yes가 아니고, y(target)이다.

smote = SMOTE(random_state=66, k_neighbors=2)
x_train, y_train = smote.fit_resample(x_train, y_train)
np.save('./_save/m30_x_train.npy', arr = x_train)

# 저장 
# import pickle
# path = './_save/'
# pickle.dump(smote, open(path + 'm30_smote_save.dat', 'wb'))
#2. 불러오기 // 모델, 훈련
import pickle
path = './_save/'
model = pickle.load(open(path + 'm30_smote_save.dat','rb'))

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
넣지 않은 것
accuracy_score : 0.9561
f1_score: 0.9521289997480473
걸린시간 :  0.08377552032470703

넣은 것
accuracy_score : 0.9474
f1_score: 0.9434523809523809
걸린시간 :  0.09296727180480957
'''

'''
accuracy_score : 0.9474
f1_score: 0.9434523809523809
걸린시간 :  0.08676815032958984

accuracy_score : 0.9561
f1_score: 0.9521289997480473
걸린시간 :  0.08320236206054688
'''