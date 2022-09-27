from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
### 스케일러 정리해놓기
### QuantileTransformer : Robust와 동일하게 이상치에 민감하지 않음


#1. 데이터
# datasets = fetch_california_housing()
datasets = load_boston()

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
# model = XGBRegressor()
model = XGBRegressor(
    n_jobs = -1,  
    n_estimators = 1000,
    learning_rate = 0.1,
)

# n_estimator : tensor에서 epoch와 동일함 


#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose = 3)
end = time.time()

result = model.score(x_test, y_test)
print('results :', result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 :', r2)
print('걸린 시간 :', end-start)
'''
results : 0.8438124187458735
r2 : 0.8438124187458735
'''
print('=======================================')
# hist = model.evals_result()
# print(hist)
'''
results : 0.8566291699938181
r2 : 0.8566291699938181
걸린 시간 : 18.44230365753174
'''