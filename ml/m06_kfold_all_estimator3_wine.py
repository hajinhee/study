from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.datasets import load_iris
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

datasets = load_wine()
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
ACC : [0.88888889 0.97222222 0.97222222 0.97142857 0.97142857] 
cross_val_score : 0.9552
AdaBoostClassifier 의 정답률 :  0.8611111111111112
BaggingClassifier 의 정답률 :  0.8611111111111112
BernoulliNB 의 정답률 :  0.4166666666666667
CalibratedClassifierCV 의 정답률 :  0.9444444444444444
CategoricalNB 은 없는 놈!!
ClassifierChain 은 없는 놈!!
ComplementNB 의 정답률 :  0.5833333333333334
DecisionTreeClassifier 의 정답률 :  0.8333333333333334
DummyClassifier 의 정답률 :  0.4166666666666667
ExtraTreeClassifier 의 정답률 :  0.8611111111111112
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  0.5555555555555556
GradientBoostingClassifier 의 정답률 :  0.8888888888888888
HistGradientBoostingClassifier 의 정답률 :  0.8888888888888888
KNeighborsClassifier 의 정답률 :  0.6388888888888888
LabelPropagation 의 정답률 :  0.3611111111111111
LabelSpreading 의 정답률 :  0.3611111111111111
LinearDiscriminantAnalysis 의 정답률 :  0.9722222222222222
LinearSVC 의 정답률 :  0.9444444444444444
LogisticRegression 의 정답률 :  0.9444444444444444
LogisticRegressionCV 의 정답률 :  0.8611111111111112
MLPClassifier 의 정답률 :  0.3611111111111111
MultiOutputClassifier 은 없는 놈!!
MultinomialNB 의 정답률 :  0.8333333333333334
NearestCentroid 의 정답률 :  0.6388888888888888
NuSVC 의 정답률 :  0.7777777777777778
OneVsOneClassifier 은 없는 놈!!
OneVsRestClassifier 은 없는 놈!!
OutputCodeClassifier 은 없는 놈!!
PassiveAggressiveClassifier 의 정답률 :  0.6111111111111112
Perceptron 의 정답률 :  0.6111111111111112
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 은 없는 놈!!
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  0.9722222222222222
RidgeClassifierCV 의 정답률 :  0.9722222222222222
SGDClassifier 의 정답률 :  0.4722222222222222
SVC 의 정답률 :  0.5555555555555556

'''