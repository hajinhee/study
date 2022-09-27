from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
datasets = fetch_covtype()
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
AdaBoostClassifier 의 정답률 :  0.4932574890493361
BaggingClassifier 의 정답률 :  0.9606808774300147
BernoulliNB 의 정답률 :  0.6307754533015498
CalibratedClassifierCV 의 정답률 :  0.7132690205932721
CategoricalNB 의 정답률 :  0.6309991996764284
ComplementNB 의 정답률 :  0.6188824729137802
DecisionTreeClassifier 의 정답률 :  0.9400015490133645
DummyClassifier 의 정답률 :  0.48700980181234566
ExtraTreeClassifier 의 정답률 :  0.8582996996635198
ExtraTreesClassifier 의 정답률 :  0.9536586835107528
GaussianNB 의 정답률 :  0.09018700033561956
GradientBoostingClassifier 의 정답률 :  0.7735858798826192
HistGradientBoostingClassifier 의 정답률 :  0.837267540424946
KNeighborsClassifier 의 정답률 :  0.9364646351643245
LinearDiscriminantAnalysis 의 정답률 :  0.6800943177026411
LinearSVC 의 정답률 :  0.7130710911078028
LogisticRegression 의 정답률 :  0.7200674681376557
LogisticRegressionCV 의 정답률 :  0.7251017615724207
MLPClassifier 의 정답률 :  0.8111150314535769
MultinomialNB 의 정답률 :  0.6395704069602334
NearestCentroid 의 정답률 :  0.38596249666531846
PassiveAggressiveClassifier 의 정답률 :  0.5954493429601645
Perceptron 의 정답률 :  0.5962066383828301
QuadraticDiscriminantAnalysis 의 정답률 :  0.09031608478266481
'''