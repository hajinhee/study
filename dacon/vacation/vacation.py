from cProfile import label
import os
from re import I, X
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pandas as pd
from icecream import ic
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from tensorflow.keras.utils import to_categorical 

# 데이터 준비
train = pd.read_csv('dacon/vacation/data/train.csv')
test = pd.read_csv('dacon/vacation/data/test.csv') 
sample_submission = pd.read_csv('dacon/vacation/data/sample_submission.csv')

# 데이터 확인
# ic(train.info())

# 데이터 시각화
# plt.hist(train.ProdTaken)
# plt.show()

# 전처리
# ic(train.isna().sum())
# ic(test.isna().sum())
# ic(sample_submission.isna().sum())

# 상관관계 분석도
plt.figure(figsize=(10,8))
heat_table = train.corr()
mask = np.zeros_like(heat_table)
mask[np.triu_indices_from(mask)] = True
heatmap_ax = sns.heatmap(heat_table, annot=True, mask = mask, cmap='coolwarm', vmin=-1, vmax=1)
heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), fontsize=10, rotation=90)
heatmap_ax.set_yticklabels(heatmap_ax.get_yticklabels(), fontsize=10)
plt.title('correlation between features', fontsize=20)
# plt.show()

# 고객의 제품 인지 방법 (회사의 홍보 or 스스로 검색) mapping
for these in [train, test]:
    these['TypeofContact'] = these['TypeofContact'].map({'Unknown': 0, 'Company Invited': 2, 'Self Enquiry': 1})

# 성별 mapping
for these in [train, test]:
    these['Gender'] = these['Gender'].map({'Male': 0, 'Female': 1, 'Fe Male': 1})

# 직업 mapping
for these in [train, test]:
    these['Occupation'] = these['Occupation'].map({'Salaried': 0, 'Small Business': 1, 'Large Business': 2, 'Free Lancer':3})

# 영업 사원이 제시한 상품 mapping
for these in [train, test]:
    these['ProductPitched'] = these['ProductPitched'].map({'Super Deluxe': 0, 'King': 1, 'Deluxe': 2, 'Standard':3, 'Basic': 4})

# 결혼 여부 mapping
for these in [train, test]:
    these['MaritalStatus'] = these['MaritalStatus'].map({'Divorced': 0, 'Married': 1, 'Unmarried': 2, 'Single':3})

# 직급 mapping
for these in [train, test]:
    these['Designation'] = these['Designation'].map({'AVP': 0, 'VP': 1, 'Manager': 2, 'Senior Manager':3, 'Executive': 4})

# 결측치 처리
ls = ['TypeofContact', 'DurationOfPitch', 'Age', 'PreferredPropertyStar']
for these in [train, test]:
    for col in ls:
        these[col].fillna(0, inplace=True)

# ic(train.corrwith(train['MonthlyIncome']))

# for these in [train, test]:
        # these['PreferredPropertyStar'].fillna(these.groupby('NumberOfTrips')['PreferredPropertyStar'].transform('mean'), inplace=True)
        # these['Age'].fillna(these['Age'].min(), inplace=True)

# 스케일링
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
train[['Age', 'DurationOfPitch']] = scaler.fit_transform(train[['Age', 'DurationOfPitch']])
test[['Age', 'DurationOfPitch']] = scaler.transform(test[['Age', 'DurationOfPitch']])

# train을 x, y로 나누고 불필요한 컬럼 제거
train.drop(columns=['id', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting', 'OwnCar',  'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1, inplace=True)
test.drop(columns=['id', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting', 'OwnCar',  'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1, inplace=True)
x = train.drop(columns=['ProdTaken'], axis=1)
y = train[['ProdTaken']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.95, shuffle=True, random_state=66)  

model = RandomForestClassifier(n_estimators=200, 
                                n_jobs=-1)
# model.fit(x, y.values.ravel())
model.fit(x_train, y_train.values.ravel())

ic('Train Accuracy : {:.2f}'.format(model.score(x_train, y_train)))
ic('Test Accuracy : {:.2f}'.format(model.score(x_test, y_test)))

y_pred = model.predict(x_test)
ic('score:', accuracy_score(y_pred, y_test)) 

y_pred = model.predict(test)
ic(np.unique(y_pred, return_counts=True))

# KFold 교차검증
# k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
# score = cross_val_score(RandomForestClassifier(), x, y, cv=k_fold, n_jobs=1, scoring='accuracy')
# ic('k_fold_score:', score) 

# 데이터 submit
y_summit = model.predict(test)
sample_submission['ProdTaken'] = y_summit
sample_submission.to_csv('dacon/vacation/save/sample_submission.csv', index=False)

