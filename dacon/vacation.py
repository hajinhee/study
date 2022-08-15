from cProfile import label
import os
from re import I, X
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pandas as pd
from icecream import ic
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold,StratifiedKFold

# 데이터 준비
train = pd.read_csv('dacon/data/train.csv')
test = pd.read_csv('dacon/data/test.csv') 
sample_submission = pd.read_csv('dacon/data/sample_submission.csv')


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

# 고객의 제품 인지 방법 (회사의 홍보 or 스스로 검색) 결측치 채우기 & 맵핑
for these in [train, test]:
    these['TypeofContact'] = these['TypeofContact'].map({'Company Invited': 0, 'Self Enquiry': 1})
    these['TypeofContact'].fillna(1, inplace=True) 

# 나이 맵핑 
age_mapping = {'Unknown': 0, 'Teenager': 1, 'Young Adult': 2, 'Adult': 3, 'Senior': 4}
bins = [-1, 0, 19, 30, 50, np.inf]  # scaling
labels = ['Unknown', 'Teenager', 'Young Adult', 'Adult', 'Senior']
for these in [train, test]:
    these['Age'] = these['Age'].fillna(-0.5)   # 결측치(누락 데이터/NaN) 치환 처리
    these['Age'] = pd.cut(these['Age'], bins, labels=labels)
    these['Age'] = these['Age'].map(age_mapping)

# 성별 맵핑
for these in [train, test]:
    these['Gender'] = these['Gender'].map({'Male': 0, 'Female': 1, 'Fe Male': 1})

# 직업 맵핑
for these in [train, test]:
    these['Occupation'] = these['Occupation'].map({'Salaried': 0, 'Small Business': 1, 'Large Business': 2, 'Free Lancer':3})

# 영업 사원이 제시한 상품 맵핑
for these in [train, test]:
    these['ProductPitched'] = these['ProductPitched'].map({'Basic': 0, 'Deluxe': 1, 'Standard': 2, 'Super Deluxe':3, 'King': 4})

# 결혼 여부 맵핑
for these in [train, test]:
    these['MaritalStatus'] = these['MaritalStatus'].map({'Married': 0, 'Divorced': 1, 'Single': 2, 'Unmarried':3})

# 직급 맵핑
for these in [train, test]:
    these['Designation'] = these['Designation'].map({'Executive': 0, 'Manager': 1, 'Senior Manager': 2, 'AVP':3, 'VP': 4})

# 영업 사원이 고객에게 제공하는 프레젠테이션 기간 맵핑
bins = [5, 15, 25, 36]
labels = [0, 1, 2]
for these in [train, test]:
    these['DurationOfPitch'] = pd.cut(these['DurationOfPitch'], bins, labels=labels) 
    these['DurationOfPitch'] = these['DurationOfPitch'].fillna(0)

# 평균 연간 여행 횟수 맵핑
bins = [1, 3, 6, 19]
labels = [0, 1, 2]
for these in [train, test]:
    these['NumberOfTrips'] = pd.cut(these['NumberOfTrips'], bins, labels=labels) 
    these['NumberOfTrips'] = these['NumberOfTrips'].fillna(0)

# 영업 사원의 프레젠테이션 후 이루어진 후속 조치 수 결측치 처리  
for these in [train, test]: 
    these['NumberOfFollowups'].fillna(4, inplace=True) 

# 선호 호텔 숙박업소 등급 결측치 처리
for these in [train, test]: 
    these['PreferredPropertyStar'].fillna(3, inplace=True) 

#  함께 여행을 계획 중인 5세 미만의 어린이 수 결측치 처리
for these in [train, test]: 
    these['NumberOfChildrenVisiting'].fillna(1, inplace=True) 

# 월급여 맵핑
ic(train['TypeofContact'].value_counts(normalize=False))
bins = [0, 20000, 35000, np.inf]
labels = [0, 1, 2]
for these in [train, test]:
    these['MonthlyIncome'] = these['MonthlyIncome'].fillna(23624.108894878707)
    these['MonthlyIncome'] = pd.cut(these['MonthlyIncome'], bins, labels=labels)


# train을 x, y로 나누고 불필요한 columns drop
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)
x = train.drop(columns=['ProdTaken'])
y = train[['ProdTaken']]

# 상관관계
print('상관관계: ', train.corr())


# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42)  

model = RandomForestClassifier()
model.fit(x, y)

result = model.score(x, y)
ic('score:', result) 


# 데이터 summit
y_summit = model.predict(test)
sample_submission['ProdTaken'] = y_summit
sample_submission.to_csv('dacon/save/sample_submission.csv', index=False)

