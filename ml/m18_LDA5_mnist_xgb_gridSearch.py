import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier, XGBRegressor
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.datasets import mnist
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print('LDA 전:',x_train.shape)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


lda = LinearDiscriminantAnalysis()
lda.fit(x_train,y_train)
x_train = lda.transform(x_train)
x_test = lda.transform(x_test)

print('LDA 후:',x_train.shape)


parameters = [
    {'n_estimators':[100, 200, 300], 'learning_rate':[0.1, 0.3, 0.001, 0.01],
    'max_depth':[4,5,6]}]

#2. 모델
from sklearn.pipeline import make_pipeline, Pipeline
model = RandomizedSearchCV(XGBClassifier(use_label_encoder=False), parameters, cv=3, verbose=3,  
                     refit=True, n_jobs=-1)

#3. 훈련
import time
start = time.time()
model.fit(x_train,y_train,eval_metric='merror')
end = time.time()


#4. 평가, 예측
result = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print('model.score :', result)
print('accuracy_score :', r2)
print('걸린 시간 :', end - start)

'''
model.score : 0.9149
accuracy_score : 0.8287100355635795
걸린 시간 : 617.4842655658722
'''