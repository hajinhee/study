
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 100)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

#2. 모델
model1 = DecisionTreeRegressor()
model2 = RandomForestRegressor()
model3 = XGBRegressor()
model4 = GradientBoostingRegressor()

#3. 훈련
model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)


#4. 평가, 예측
result = model1.score(x_test, y_test)
result = model2.score(x_test, y_test)
result = model3.score(x_test, y_test)
result = model4.score(x_test, y_test)

# from sklearn.metrics import accuracy_score
# y_predict = model1.predict(x_test)
# acc = mode(y_test, y_predict)

# print('accuracy_score :', acc)
# print(model1.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align = 'center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    
plt.figure(figsize=(22,20))
plt.subplot(2, 2, 1)
plot_feature_importance_dataset(model1)
plt.subplot(2, 2, 2)
plot_feature_importance_dataset(model2)
plt.subplot(2, 2, 3)
plot_feature_importance_dataset(model3)
plt.subplot(2, 2, 4)
plot_feature_importance_dataset(model4)
plt.show()

print(model1.feature_importances_)
print(model2.feature_importances_)
print(model3.feature_importances_)
print(model4.feature_importances_)