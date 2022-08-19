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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import matplotlib.pyplot as plt
import seaborn as sns


# 데이터 준비
train = pd.read_csv('dacon/vacation/data/train.csv')
test = pd.read_csv('dacon/vacation/data/test.csv') 
sample_submission = pd.read_csv('dacon/vacation/data/sample_submission.csv')

# 결측치 처리
value = 0
def handle_na(data):
    temp = data.copy()
    for col, dtype in temp.dtypes.items():
        if dtype == 'object':
            # 문자형 칼럼의 경우 'Unknown'을 채워줍니다.
            global value
            value = 'Unknown'
        elif dtype == int or dtype == float:
            # 수치형 칼럼의 경우 0을 채워줍니다.
            value = 0
        temp.loc[:,col] = temp[col].fillna(value)
    return temp

train_nona = handle_na(train)
test = handle_na(test)

encoder = LabelEncoder()
encoder.fit(train_nona['TypeofContact'])
encoder.transform(train_nona['TypeofContact'])

train_enc = train_nona.copy()

# LabelEncoder
object_columns = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_enc[o_col])
    train_enc[o_col] = encoder.transform(train_enc[o_col])

for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_nona[o_col])
    test[o_col] = encoder.transform(test[o_col])


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


# 스케일링
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = MaxAbsScaler()
train[['Age', 'DurationOfPitch']] = scaler.fit_transform(train[['Age', 'DurationOfPitch']])
test[['Age', 'DurationOfPitch']] = scaler.transform(test[['Age', 'DurationOfPitch']])


# 불필요한 컬럼 제거
train = train_enc.drop(columns=['id','NumberOfChildrenVisiting', 'NumberOfPersonVisiting', 'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
test = test.drop(columns=['id', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting', 'OwnCar', 'MonthlyIncome', 'NumberOfTrips', 'NumberOfFollowups'], axis=1)
x = train.drop(columns=['ProdTaken'], axis=1)
y = train[['ProdTaken']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=66)  

model = RandomForestClassifier(n_estimators=200, 
                                random_state=0,
                                n_jobs=-1)
model.fit(x_train, y_train.values.ravel())
# model.fit(x, y.values.ravel())

ic('Train Accuracy : {:.2f}'.format(model.score(x_train, y_train)))
ic('Test Accuracy : {:.2f}'.format(model.score(x_test, y_test)))

y_pred = model.predict(x_test)
ic('score:', accuracy_score(y_pred, y_test)) 

y_pred = model.predict(test)
ic(np.unique(y_pred, return_counts=True))


# 데이터 submit
y_summit = model.predict(test)
sample_submission['ProdTaken'] = y_summit
sample_submission.to_csv('dacon/vacation/save/sample_submission.csv', index=False)
