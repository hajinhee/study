from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer


#1. 데이터
datasets = load_breast_cancer()
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
AdaBoostClassifier 의 정답률 :  0.9473684210526315
BaggingClassifier 의 정답률 :  0.9736842105263158
BernoulliNB 의 정답률 :  0.5614035087719298
CalibratedClassifierCV 의 정답률 :  0.9649122807017544
ComplementNB 의 정답률 :  0.8508771929824561
DecisionTreeClassifier 의 정답률 :  0.956140350877193
DummyClassifier 의 정답률 :  0.5701754385964912
ExtraTreeClassifier 의 정답률 :  0.9473684210526315
ExtraTreesClassifier 의 정답률 :  0.956140350877193
GaussianNB 의 정답률 :  0.9385964912280702
GaussianProcessClassifier 의 정답률 :  0.9649122807017544
GradientBoostingClassifier 의 정답률 :  0.9649122807017544
HistGradientBoostingClassifier 의 정답률 :  0.956140350877193
KNeighborsClassifier 의 정답률 :  0.956140350877193
LabelPropagation 의 정답률 :  0.9649122807017544
LabelSpreading 의 정답률 :  0.9649122807017544
LinearDiscriminantAnalysis 의 정답률 :  0.956140350877193
LinearSVC 의 정답률 :  0.9649122807017544
LogisticRegression 의 정답률 :  0.9649122807017544
LogisticRegressionCV 의 정답률 :  0.9649122807017544
MLPClassifier 의 정답률 :  0.9649122807017544
MultinomialNB 의 정답률 :  0.8157894736842105
NearestCentroid 의 정답률 :  0.9298245614035088
NuSVC 의 정답률 :  0.9385964912280702
PassiveAggressiveClassifier 의 정답률 :  0.9649122807017544
Perceptron 의 정답률 :  0.8771929824561403
QuadraticDiscriminantAnalysis 의 정답률 :  0.956140350877193
RandomForestClassifier 의 정답률 :  0.956140350877193
RidgeClassifier 의 정답률 :  0.9473684210526315
RidgeClassifierCV 의 정답률 :  0.9473684210526315
SGDClassifier 의 정답률 :  0.9473684210526315
SVC 의 정답률 :  0.9649122807017544
'''