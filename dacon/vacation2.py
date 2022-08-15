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
from sklearn.preprocessing import LabelEncoder


# 데이터 준비
train = pd.read_csv('dacon/data/train.csv')
test = pd.read_csv('dacon/data/test.csv') 
sample_submission = pd.read_csv('dacon/data/sample_submission.csv')

# 결측치를 처리하는 함수를 작성합니다.
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

encoder = LabelEncoder()
encoder.fit(train_nona['TypeofContact'])
encoder.transform(train_nona['TypeofContact'])

train_enc = train_nona.copy()

# 모든 문자형 변수에 대해 encoder를 적용합니다.
object_columns = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_enc[o_col])
    train_enc[o_col] = encoder.transform(train_enc[o_col])

# 결측치 처리
test = handle_na(test)

# 문자형 변수 전처리
for o_col in object_columns:
    encoder = LabelEncoder()
    encoder.fit(train_nona[o_col])
    test[o_col] = encoder.transform(test[o_col])


# 분석할 의미가 없는 칼럼을 제거합니다.
train = train_enc.drop(columns=['id'])
test = test.drop(columns=['id'])

# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
x_train = train.drop(columns=['ProdTaken'])
y_train = train[['ProdTaken']]

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=42)  

model = RandomForestClassifier()
model.fit(x_train, y_train.values.ravel())
# model.fit(x, y.values.ravel())

# ic('Train Accuracy : {:.2f}'.format(model.score(x_train, y_train)))
# ic('Test Accuracy : {:.2f}'.format(model.score(x_test, y_test)))

# y_pred = model.predict(x_test)
# ic('score:', accuracy_score(y_pred, y_test)) 


# 데이터 submit
y_summit = model.predict(test)
sample_submission['ProdTaken'] = y_summit
sample_submission.to_csv('dacon/save/sample_submission.csv', index=False)
