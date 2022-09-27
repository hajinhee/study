import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import time
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

#1. 데이터
path = 'D:\\_data\\'
datasets = pd.read_csv(path + 'winequality-white.csv',sep=';', header = 0)

count_data = datasets.groupby("quality")['quality'].count()  
print(count_data)

plt.bar(count_data.index, count_data)
plt.show()

# gd = datasets.groupby( [ "quality"] ).count()  
# gd.plot(kind='bar', rot=0)
# plt.show()

# null = np.sum(pd.isnull(datasets))
# print(null)

#########그래프 그리기########

p = pd.DataFrame({'count' : datasets.groupby( [ "quality"] ).size()})
print(p)
p.plot(kind='bar', rot=0)