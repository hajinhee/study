from cProfile import label
from re import I, X
import os
import sys
from tabnanny import verbose
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import pandas as pd
from icecream import ic
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold  # 회귀: KFold, 분류: StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from bayes_opt import BayesianOptimization
from sklearn.metrics import matthews_corrcoef
from sklearn.impute import SimpleImputer

# 데이터 준비
train = pd.read_csv('dacon/travel/data/train.csv')
test = pd.read_csv('dacon/travel/data/test.csv') 
sample_submission = pd.read_csv('dacon/travel/data/sample_submission.csv')

# 데이터 확인
# ic(train.info())
# ic(train.describe())  # 수치형 예측변수 요약
# ic(train.describe(include='object'))  # 문자형 예측변수 요약

# 데이터 시각화
# plt.hist(train.ProdTaken)
# plt.show()

# 전처리
# ic(test.isna().sum())
# ic(sample_submission.isna().sum())

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

# 상관관계 분석도
# plt.figure(figsize=(10, 8))
# heat_table = train.corr()
# mask = np.zeros_like(heat_table)
# mask[np.triu_indices_from(mask)] = True
# heatmap_ax = sns.heatmap(heat_table, annot=True, mask = mask, cmap='coolwarm', vmin=-1, vmax=1)
# heatmap_ax.set_xticklabels(heatmap_ax.get_xticklabels(), fontsize=10, rotation=90)
# heatmap_ax.set_yticklabels(heatmap_ax.get_yticklabels(), fontsize=10)
# plt.title('correlation between features', fontsize=20)
# plt.show()

plt.figure(figsize=(8, 4))
sns.countplot(x='OwnCar', hue='ProdTaken', data=train).set(title='OwnCar')

# 이상치 확인
# plt.scatter(train.NumberOfFollowups, train.NumberOfPersonVisiting)
# plt.xlabel('NumberOfFollowups')
# plt.ylabel('NumberOfPersonVisiting')
# plt.show()

# 이상치 제거
# ic(train['NumberOfFollowups'].sort_values(ascending=False), '\n')
# ic(train.iloc[[189]])
train.drop(index=[189, 604, 1338, 987], inplace=True)

# 결측치 처리
ls = ['DurationOfPitch', 'TypeofContact']
for these in [train, test]:
    for col in ls:
        these[col].fillna(0, inplace=True) 

for these in [train, test]:
        these['PreferredPropertyStar'].fillna(these.groupby('NumberOfTrips')['PreferredPropertyStar'].transform('mean'), inplace=True)
        these['NumberOfTrips'].fillna(these.groupby('DurationOfPitch')['NumberOfTrips'].transform('mean'), inplace=True)
        these['Age'].fillna(these.groupby('Designation')['Age'].transform('mean'), inplace=True)

# 스케일
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = MaxAbsScaler()
train[['Age', 'DurationOfPitch']] = scaler.fit_transform(train[['Age', 'DurationOfPitch']])
test[['Age', 'DurationOfPitch']] = scaler.transform(test[['Age', 'DurationOfPitch']])
# ic(train.describe())  

# train을 x, y로 나누고 불필요한 컬럼 제거
train.drop(columns=['id', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting', 'OwnCar', 'MonthlyIncome', 'NumberOfFollowups'], axis=1, inplace=True)
test.drop(columns=['id', 'NumberOfChildrenVisiting', 'NumberOfPersonVisiting', 'OwnCar', 'MonthlyIncome', 'NumberOfFollowups'], axis=1, inplace=True)
x = train.drop(columns=['ProdTaken'], axis=1)
y = train[['ProdTaken']]

k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # StratifiedKFold --> 분류문제에 사용
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=1234, stratify=y)  # stratify=y -> y개수만큼 컷

#  BayesianOptimization
def CB_opt(n_estimators, depth, learning_rate, 
            l2_leaf_reg, model_size_reg, od_pval): 
    scores = []
    reg = CatBoostClassifier(verbose=0,
                            n_estimators=int(n_estimators),
                            learning_rate=learning_rate,
                            l2_leaf_reg=l2_leaf_reg,
                            max_depth=int(depth),
                            random_state=42,
                            grow_policy='Lossguide',
                            use_best_model=True, 
                            model_size_reg=model_size_reg,
                            od_pval=od_pval
                            )
    reg.fit(x_train, y_train, eval_set=(x_test, y_test))
    scores.append(matthews_corrcoef(y_test, reg.predict(x_test)))
    return np.mean(scores)

pbounds = {'n_estimators': (150, 1000),
           'depth': (4, 12),
           'learning_rate': (.01, 0.3),
           'l2_leaf_reg': (0, 10),
           'model_size_reg': (0, 10),
           'od_pval' : (0, 5)
}
optimizer = BayesianOptimization(
    f=CB_opt,
    pbounds=pbounds,
    verbose=0,
    random_state=42,
)
optimizer.maximize(init_points=2, n_iter=20)
ic(optimizer.max)

cat_params = {
    # 'depth': [10, 11, 14, 15],
    'l2_leaf_reg': [9.122611898980937],    
    'learning_rate': [0.279118773316628],
    'model_size_reg': [5.669787571831169],
    # 'n_estimators': [200, 300, 400, 500, 600],
    'od_pval': [0.0],
}
                             
cat = CatBoostClassifier(verbose=2, depth=14, n_estimators=200, allow_writing_files=False)  
grid_cv = GridSearchCV(cat,
                       param_grid=cat_params,
                       cv=k_fold,
                       n_jobs=-1)

grid_cv.fit(x, y.values.ravel())
ic(grid_cv.best_estimator_, grid_cv.best_params_, grid_cv.best_score_)  # best_score_: 최고 점수, best_params_: 최고 점수를 낸 파라미터, best_estimator_: 최고 점수를 낸 파라미터를 가진 모형

model = grid_cv.best_estimator_
model.fit(x, y.values.ravel())

# model.fit(x_train, y_train.values.ravel())
# y_pred = model.predict(x_test)
# ic('score:', accuracy_score(y_pred, y_test)) 

# KFold 교차검증
# score = cross_val_score(model, x, y.values.ravel(), cv=k_fold, n_jobs=-1, scoring='accuracy')
# ic(score, np.mean(score)) 

# 데이터 submit
y_summit = model.predict(test)
ic(np.unique(y, return_counts=True))
ic(np.unique(y_summit, return_counts=True))
sample_submission['ProdTaken'] = y_summit
sample_submission.to_csv('dacon/travel/save/sample_submission.csv', index=False)
