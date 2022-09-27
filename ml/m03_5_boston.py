from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=100)


#2. 모델 구성
allAlgorithms = all_estimators(type_filter = 'classifier')
# allAlgorithms = all_estimators(type_filter = 'regressor')  
# allAlgorithms XGBoost, Catboost, LGBM은 없다. >> 
print('allAlgorithms :', allAlgorithms)
print('모델의 갯수 :', len(allAlgorithms))

for (name, algorithms) in allAlgorithms:
    try:
        model = algorithms()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)
    except:
        continue

# #3. 훈련
# model.fit(x_train,y_train)

# #4. 평가, 예측
# result = model.score(x_test, y_test)

# from sklearn.metrics import accuracy_score
# from sklearn.metrics import r2_score
# y_predict = model.predict(x_test)

# # acc = accuracy_score(y_test, y_predict)
# loss = model.(x_test, y_test)
# r2 = r2_score(x_test, y_test)
