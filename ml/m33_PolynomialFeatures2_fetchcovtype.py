import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

datasets = fetch_covtype()
x = datasets.data
y = datasets.target

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
xp = pf.fit_transform(x)
print(xp.shape)

x_train, x_test, y_train, y_test = train_test_split(xp,y,
        train_size =0.8, shuffle=True, random_state = 100)

#2. 모델구성
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

model = make_pipeline(StandardScaler(), LGBMClassifier())

model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('accuracy_score :', acc)
