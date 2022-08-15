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

# # 나이 mapping 
# age_mapping = {'Unknown': 0, 'Teenager': 1, 'Young Adult': 2, 'Adult': 3, 'Senior': 4}
# bins = [-1, 0, 19, 30, 50, np.inf]  # scaling
# labels = ['Unknown', 'Teenager', 'Young Adult', 'Adult', 'Senior']
# for these in [train, test]:
#     these['Age'] = pd.cut(these['Age'], bins, labels=labels)
#     these['Age'] = these['Age'].map(age_mapping)

# 고객의 제품 인지 방법 (회사의 홍보 or 스스로 검색) mapping
for these in [train, test]:
    these['TypeofContact'] = these['TypeofContact'].map({'Unknown': 0, 'Company Invited': 1, 'Self Enquiry': 2})

# 성별 mapping
for these in [train, test]:
    these['Gender'] = these['Gender'].map({'Male': 0, 'Female': 1, 'Fe Male': 1})

# 직업 mapping
for these in [train, test]:
    these['Occupation'] = these['Occupation'].map({'Salaried': 0, 'Small Business': 1, 'Large Business': 2, 'Free Lancer':3})

# 영업 사원이 제시한 상품 mapping
for these in [train, test]:
    these['ProductPitched'] = these['ProductPitched'].map({'Basic': 0, 'Deluxe': 1, 'Standard': 2, 'Super Deluxe':3, 'King': 4})

# 결혼 여부 mapping
for these in [train, test]:
    these['MaritalStatus'] = these['MaritalStatus'].map({'Married': 0, 'Divorced': 1, 'Single': 2, 'Unmarried':3})

# 직급 mapping
for these in [train, test]:
    these['Designation'] = these['Designation'].map({'Executive': 0, 'Manager': 1, 'Senior Manager': 2, 'AVP':3, 'VP': 4})

# # 영업 사원이 고객에게 제공하는 프레젠테이션 기간 mapping
# bins = [0, 5, 15, 25, 36]
# labels = [0, 1, 2, 3]
# for these in [train, test]:
#     these['DurationOfPitch'] = pd.cut(these['DurationOfPitch'], bins, labels=labels)

# # 평균 연간 여행 횟수 mapping
# bins = [1, 3, 6, 19]
# labels = [0, 1, 2]
# for these in [train, test]:
#     these['NumberOfTrips'] = pd.cut(these['NumberOfTrips'], bins, labels=labels) 

# 월급여 mapping
# bins = [0, 20000, 25000, 35000, np.inf]
# labels = [0, 1, 2, 3]
# for these in [train, test]:
#     these['MonthlyIncome'] = pd.cut(these['MonthlyIncome'], bins, labels=labels)

# 결측치 제거
ls = ['TypeofContact', 'Age', 'MonthlyIncome', 'DurationOfPitch', 'NumberOfTrips', 'NumberOfFollowups',
'PreferredPropertyStar', 'NumberOfChildrenVisiting']
for these in [train, test]:
    for i in ls:
        these[i].fillna(0, inplace=True)


# train을 x, y로 나누고 불필요한 컬럼 제거
train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)
x = train.drop(columns=['ProdTaken'])
y = train[['ProdTaken']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42)  


model = RandomForestClassifier()
model.fit(x_train, y_train)

ic('Train Accuracy : {:.2f}'.format(model.score(x_train, y_train)))
ic('Test Accuracy : {:.2f}'.format(model.score(x_test, y_test)))

y_pred = model.predict(x_test)
ic('score:', accuracy_score(y_pred, y_test)) 


# 데이터 submit
y_summit = model.predict(test)
sample_submission['ProdTaken'] = y_summit
sample_submission.to_csv('dacon/save/sample_submission.csv', index=False)

