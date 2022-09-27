from sklearn.experimental import enable_halving_search_cv
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.svm import LinearSVC, SVC
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

#1. 데이터

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 100)

from sklearn.decomposition import PCA  # 컬럼을 압축하다.

# parameters = [
#     {'randomforestclassifier__max_depth' : [6,8,10],
#     'randomforestclassifier__min_samples_leaf' :[3,5,7],
#      'randomforestclassifier__min_samples_split' : [3,5,10]}]

parameters = [
    {'rf__max_depth' : [6,8,10],
    'rf__min_samples_leaf' :[3,5,7],
     'rf__min_samples_split' : [3,5,10]}]

#2. 모델
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# pipe = make_pipeline(MinMaxScaler(),RandomForestClassifier())
pipe = Pipeline([('mm',MinMaxScaler()),('rf', RandomForestClassifier())])
model = GridSearchCV(pipe, parameters, cv = 5, verbose = 3)
# model = RandomizedSearchCV(pipe, parameters, cv = 5, verbose = 3)
# model = HalvingGridSearchCV(pipe, parameters, cv = 5, verbose = 3)

# pipe의 파라미터를 넣어줘야 한다. 

#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train)
end = time.time()

#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print('model.score :', result)
print('accuracy_score :', acc)
print('걸린 시간 :', end - start)


'''
- grid 
model.score : 0.9666666666666667
accuracy_score : 0.9666666666666667
걸린 시간 : 11.884044885635376

- Randomized
model.score : 0.9666666666666667
accuracy_score : 0.9666666666666667
걸린 시간 : 4.4096550941467285

- halving
model.score : 0.9666666666666667
accuracy_score : 0.9666666666666667
걸린 시간 : 17.001869440078735
'''