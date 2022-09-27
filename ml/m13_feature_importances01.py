import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 100)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#2. 모델
# model = DecisionTreeClassifier(max_depth=5)
# model = RandomForestClassifier(max_depth=5)
# model = XGBClassifier()
model = GradientBoostingClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('accuracy_score :', acc)

print(model.feature_importances_)

# [0.01253395 0.01880092 0.5733795  0.39528563]
# [0.10737758 0.02798842 0.50036975 0.36426424]
# [0.03014198 0.02563514 0.6144062  0.32981667]
# [0.00555117 0.01351838 0.35995639 0.62097406]