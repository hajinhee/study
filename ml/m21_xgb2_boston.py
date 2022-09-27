from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_california_housing()
# datasets = load_boston()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)   # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle = True, random_state = 66, train_size=0.8
)

scaler = QuantileTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBRegressor(
    n_jobs = -1,  
    n_estimators = 2000,
    learning_rate = 0.025,
    max_depth = 4,
    min_child_weight = 1,
    subsample = 1,
    colsample_bytree = 1,
    reg_alpha = 1,              # 규제 : L1
    reg_lamda = 0,              # 규제 : L2
    
)

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose = 3,
          eval_set=[(x_train,y_train),(x_test, y_test)],
          eval_metric='mae',              #rmse, mae, logloss, error
          )
end = time.time()

result = model.score(x_test, y_test)
print('results :', round(result,4))

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 :',round(r2,4))
print('걸린 시간 :', round(end-start, 4))

print('=======================================')
hist = model.evals_result()
print(hist)

import matplotlib.pyplot as plt

loss1 = hist.get('validation_0').get('mae')
loss2 = hist.get('validation_1').get('mae')
plt.plot(loss1, 'y--', label="training loss")
plt.plot(loss2, 'r--', label="test loss")
plt.grid()
plt.legend()
plt.show()