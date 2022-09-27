import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import matplotlib.pyplot as plt
from pprint import pprint

### Series : pandas에서 한 개 있는것  // DataFrame은 두 개 이상
### vector : numpy에서 한 개 있는 것

#1. 데이터
path = 'D:\\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv',sep=';'
                       , header = 0)
'''
count_data = datasets.groupby("quality")['quality'].count()  
pprint(count_data)

# plt.bar(count_data.index, count_data)
# plt.show()
x = datasets.drop("quality",axis=1)
y = datasets["quality"]
pprint(x.shape)

# if를 사용해서 y의 라벨 값을 수정해서 

### 필수로 외우기 
y = np.where(y == 9, 8, y)
y = np.where(y == 7, 8, y)
y = np.where(y == 6, 4, y)
y = np.where(y == 5, 4, y)
y = np.where(y == 3, 1, y)
y = np.where(y == 2, 1, y)

'''
##### 선생님 방법 #####
datasets = datasets.values
x = datasets[:,:11]
y = datasets[:,11]

newlist = []

for i in y:
      # print(i)
      if i <= 4:
            newlist += [0]
      elif  i <= 7:
            newlist += [1]
      else:
            newlist += [2]
            
y = np.array(newlist)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([ 183, 4535,  180], dtype=int64))

# print(np.unique(y))
# pprint(np.unique(y, return_counts = True))

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


'''
results : 0.9286
r2 : 0.9286
걸린 시간 : 0.5476

정리 : 권한이 있는 상황에서 y 갑의 라벨을 줄인다면, 
'''