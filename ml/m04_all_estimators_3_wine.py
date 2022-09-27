from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=100)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

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

'''
AdaBoostClassifier 의 정답률 :  0.8611111111111112
BaggingClassifier 의 정답률 :  0.8333333333333334
BernoulliNB 의 정답률 :  0.3333333333333333
CalibratedClassifierCV 의 정답률 :  0.9722222222222222
CategoricalNB 의 정답률 :  0.5
ComplementNB 의 정답률 :  0.8333333333333334
DecisionTreeClassifier 의 정답률 :  0.8333333333333334
DummyClassifier 의 정답률 :  0.4166666666666667
ExtraTreeClassifier 의 정답률 :  0.9722222222222222
ExtraTreesClassifier 의 정답률 :  0.9722222222222222
GaussianNB 의 정답률 :  1.0
GaussianProcessClassifier 의 정답률 :  0.9722222222222222
GradientBoostingClassifier 의 정답률 :  0.8888888888888888
HistGradientBoostingClassifier 의 정답률 :  0.8888888888888888
KNeighborsClassifier 의 정답률 :  0.8888888888888888
LabelPropagation 의 정답률 :  0.9166666666666666
LabelSpreading 의 정답률 :  0.9166666666666666
LinearDiscriminantAnalysis 의 정답률 :  0.9722222222222222
LinearSVC 의 정답률 :  0.9722222222222222
LogisticRegression 의 정답률 :  0.9722222222222222
LogisticRegressionCV 의 정답률 :  0.9722222222222222
MLPClassifier 의 정답률 :  0.9166666666666666
MultinomialNB 의 정답률 :  0.9166666666666666
NearestCentroid 의 정답률 :  0.9444444444444444
NuSVC 의 정답률 :  0.9444444444444444
PassiveAggressiveClassifier 의 정답률 :  0.9444444444444444
Perceptron 의 정답률 :  0.9722222222222222
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 의 정답률 :  0.9444444444444444
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  0.9722222222222222
RidgeClassifierCV 의 정답률 :  0.9722222222222222
SGDClassifier 의 정답률 :  0.9722222222222222
SVC 의 정답률 :  0.9722222222222222
'''