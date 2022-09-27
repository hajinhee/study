from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size =0.8, shuffle=True, random_state = 100)

from sklearn.model_selection import train_test_split, KFold, cross_val_score

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state=100)

model = XGBClassifier()

scores = cross_val_score(model, x, y, cv = kfold)
print('ACC :',scores, "\ncross_val_score :", round(np.mean(scores),4))


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
        # continue
        print(name,'은 없는 놈!!')

'''
ACC : [0.95614035 0.95614035 0.99122807 0.94736842 0.95575221] 
cross_val_score : 0.9613
AdaBoostClassifier 의 정답률 :  0.9473684210526315
BaggingClassifier 의 정답률 :  0.9473684210526315
BernoulliNB 의 정답률 :  0.5701754385964912
CalibratedClassifierCV 의 정답률 :  0.9473684210526315
CategoricalNB 은 없는 놈!!
ClassifierChain 은 없는 놈!!
ComplementNB 의 정답률 :  0.9122807017543859
DecisionTreeClassifier 의 정답률 :  0.956140350877193
DummyClassifier 의 정답률 :  0.5701754385964912
ExtraTreeClassifier 의 정답률 :  0.9298245614035088
ExtraTreesClassifier 의 정답률 :  0.956140350877193
GaussianNB 의 정답률 :  0.9298245614035088
GaussianProcessClassifier 의 정답률 :  0.9298245614035088
GradientBoostingClassifier 의 정답률 :  0.956140350877193
HistGradientBoostingClassifier 의 정답률 :  0.956140350877193
KNeighborsClassifier 의 정답률 :  0.9473684210526315
LabelPropagation 의 정답률 :  0.4473684210526316
LabelSpreading 의 정답률 :  0.4473684210526316
LinearDiscriminantAnalysis 의 정답률 :  0.956140350877193
LinearSVC 의 정답률 :  0.956140350877193
LogisticRegression 의 정답률 :  0.956140350877193
LogisticRegressionCV 의 정답률 :  0.956140350877193
MLPClassifier 의 정답률 :  0.9122807017543859
MultiOutputClassifier 은 없는 놈!!
MultinomialNB 의 정답률 :  0.9122807017543859
NearestCentroid 의 정답률 :  0.9210526315789473
NuSVC 의 정답률 :  0.8508771929824561
OneVsOneClassifier 은 없는 놈!!
OneVsRestClassifier 은 없는 놈!!
OutputCodeClassifier 은 없는 놈!!
PassiveAggressiveClassifier 의 정답률 :  0.8859649122807017
Perceptron 의 정답률 :  0.9473684210526315
QuadraticDiscriminantAnalysis 의 정답률 :  0.956140350877193
RadiusNeighborsClassifier 은 없는 놈!!
RandomForestClassifier 의 정답률 :  0.9736842105263158
RidgeClassifier 의 정답률 :  0.956140350877193
RidgeClassifierCV 의 정답률 :  0.9473684210526315
SGDClassifier 의 정답률 :  0.9298245614035088
SVC 의 정답률 :  0.9473684210526315
StackingClassifier 은 없는 놈!!
VotingClassifier 은 없는 놈!!
'''