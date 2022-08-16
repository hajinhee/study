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



# 데이터 준비
train = pd.read_csv('dacon/vacation/data/train.csv')
test = pd.read_csv('dacon/vacation/data/test.csv') 
sample_submission = pd.read_csv('dacon/vacation/data/sample_submission.csv')


# 데이터 확인
# ic(train.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1955 entries, 0 to 1954
Data columns (total 20 columns):
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   id                        1955 non-null   int64
 1   Age                       1861 non-null   float64
 2   TypeofContact             1945 non-null   object
 3   CityTier                  1955 non-null   int64
 4   DurationOfPitch           1853 non-null   float64
 5   Occupation                1955 non-null   object
 6   Gender                    1955 non-null   object
 7   NumberOfPersonVisiting    1955 non-null   int64
 8   NumberOfFollowups         1942 non-null   float64
 9   ProductPitched            1955 non-null   object
 10  PreferredPropertyStar     1945 non-null   float64
 11  MaritalStatus             1955 non-null   object
 12  NumberOfTrips             1898 non-null   float64
 13  Passport                  1955 non-null   int64
 14  PitchSatisfactionScore    1955 non-null   int64
 15  OwnCar                    1955 non-null   int64
 16  NumberOfChildrenVisiting  1928 non-null   float64
 17  Designation               1955 non-null   object
 18  MonthlyIncome             1855 non-null   float64
 19  ProdTaken                 1955 non-null   int64
dtypes: float64(7), int64(7), object(6)
'''
# ic(test.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2933 entries, 0 to 2932
Data columns (total 19 columns):
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   id                        2933 non-null   int64
 1   Age                       2801 non-null   float64
 2   TypeofContact             2918 non-null   object
 3   CityTier                  2933 non-null   int64
 4   DurationOfPitch           2784 non-null   float64
 5   Occupation                2933 non-null   object
 6   Gender                    2933 non-null   object
 7   NumberOfPersonVisiting    2933 non-null   int64
 8   NumberOfFollowups         2901 non-null   float64
 9   ProductPitched            2933 non-null   object
 10  PreferredPropertyStar     2917 non-null   float64
 11  MaritalStatus             2933 non-null   object
 12  NumberOfTrips             2850 non-null   float64
 13  Passport                  2933 non-null   int64
 14  PitchSatisfactionScore    2933 non-null   int64
 15  OwnCar                    2933 non-null   int64
 16  NumberOfChildrenVisiting  2894 non-null   float64
 17  Designation               2933 non-null   object
 18  MonthlyIncome             2800 non-null   float64
dtypes: float64(7), int64(6), object(6)
'''
# ic(sample_submission.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2933 entries, 0 to 2932
Data columns (total 2 columns):
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   id         2933 non-null   int64
 1   ProdTaken  2933 non-null   int64
dtypes: int64(2)
'''

# 데이터 시각화
# plt.hist(train.ProdTaken)
# plt.show()


# 전처리
# ic(train.isna().sum())
'''
Age                          94
TypeofContact                10
CityTier                      0
DurationOfPitch             102
Occupation                    0
Gender                        0
NumberOfPersonVisiting        0
NumberOfFollowups            13
ProductPitched                0
PreferredPropertyStar        10
MaritalStatus                 0
NumberOfTrips                57
Passport                      0
PitchSatisfactionScore        0
OwnCar                        0
NumberOfChildrenVisiting     27
Designation                   0
MonthlyIncome               100
ProdTaken                     0
'''
# ic(test.isna().sum())
'''
Age                         132
TypeofContact                15
CityTier                      0
DurationOfPitch             149
Occupation                    0
Gender                        0
NumberOfPersonVisiting        0
NumberOfFollowups            32
ProductPitched                0
PreferredPropertyStar        16
MaritalStatus                 0
NumberOfTrips                83
Passport                      0
PitchSatisfactionScore        0
OwnCar                        0
NumberOfChildrenVisiting     39
Designation                   0
MonthlyIncome               133
'''
# ic(sample_submission.isna().sum())
'''
ProdTaken    0
'''

#상관관계 분석도
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
ls = ['TypeofContact', 'DurationOfPitch', ]
for these in [train, test]:
    for col in ls:
        these[col].fillna(0, inplace=True)

mean_cols = ['Age','NumberOfFollowups','PreferredPropertyStar', 'MonthlyIncome',
            'NumberOfTrips','NumberOfChildrenVisiting']
for these in [train, test]:
    for col in mean_cols:
        these[col] = these[col].fillna(train[col].mean())

# 스케일링
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
train[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.fit_transform(train[['Age', 'DurationOfPitch', 'MonthlyIncome']])
test[['Age', 'DurationOfPitch', 'MonthlyIncome']] = scaler.transform(test[['Age', 'DurationOfPitch', 'MonthlyIncome']])

# train을 x, y로 나누고 불필요한 컬럼 제거
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)
x = train.drop(columns=['ProdTaken'])
y = train[['ProdTaken']]

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42)  

model = RandomForestClassifier()
model.fit(x, y.values.ravel())

# ic('Train Accuracy : {:.2f}'.format(model.score(x_train, y_train)))
# ic('Test Accuracy : {:.2f}'.format(model.score(x_test, y_test)))

# y_pred = model.predict(x_test)
# ic('score:', accuracy_score(y_pred, y_test)) 


# 데이터 submit
y_summit = model.predict(test)
sample_submission['ProdTaken'] = y_summit
sample_submission.to_csv('dacon/vacation/save/sample_submission.csv', index=False)

