import sklearn
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


datasets = load_boston()
# datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(datasets.feature_names)
print(datasets.DESCR)
print(x.shape, y.shape) 

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.1, random_state=66 )

# model = LinearRegression()
model = make_pipeline(StandardScaler(), LinearRegression())

model.fit(x_train, y_train)
print(model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2')  # scoring 어떤평가지표를 사용할 것인가
print(scores)
print(sklearn.metrics.SCORERS.keys())

##################PolynomialFeature 후

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
xp = pf.fit_transform(x)
print(xp.shape)

x_train, x_test, y_train, y_test = train_test_split(
    xp, y, test_size = 0.1, random_state=66 )

# model = LinearRegression()
model = make_pipeline(StandardScaler(), LinearRegression())

model.fit(x_train, y_train)
print(model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2')  # scoring 어떤평가지표를 사용할 것인가
print(scores)
'''
(506, 13) (506,)  >>  (506, 105)
feature가 엄청나게 증가했다. 
y = w1x1+w2x2+w3x3 를 y = w제곱 + x1x2
1차함수의 데이터의 구조를 제곱해서 
'''