import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from icecream import ic

'''
평균 제곱근 편차(Root Mean Square Deviation; RMSD) 또는 평균 제곱근 오차(Root Mean Square Error; RMSE)는 
추정 값 또는 모델이 예측한 값과 실제 환경에서 관찰되는 값의 차이를 다룰 때 흔히 사용하는 측도이다.
'''
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) 

#1. 데이터 로드 및 정제
path = "./data/bike/" 

train = pd.read_csv(path + 'train.csv')              
test = pd.read_csv(path + 'test.csv')                  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')     

ic(train.info())
ic(train.columns)  # ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10886 entries, 0 to 10885
Data columns (total 12 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   datetime    10886 non-null  object
 1   season      10886 non-null  int64
 2   holiday     10886 non-null  int64
 3   workingday  10886 non-null  int64
 4   weather     10886 non-null  int64
 5   temp        10886 non-null  float64
 6   atemp       10886 non-null  float64
 7   humidity    10886 non-null  int64
 8   windspeed   10886 non-null  float64
 9   casual      10886 non-null  int64
 10  registered  10886 non-null  int64
 11  count       10886 non-null  int64
dtypes: float64(3), int64(8), object(1)
'''
ic(test.info())
ic(test.columns)  # ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6493 entries, 0 to 6492
Data columns (total 9 columns):
 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   datetime    6493 non-null   object
 1   season      6493 non-null   int64
 2   holiday     6493 non-null   int64
 3   workingday  6493 non-null   int64
 4   weather     6493 non-null   int64
 5   temp        6493 non-null   float64
 6   atemp       6493 non-null   float64
 7   humidity    6493 non-null   int64
 8   windspeed   6493 non-null   float64
dtypes: float64(3), int64(5), object(1)
'''
ic(submit_file.info())
ic(submit_file.columns)  # ['datetime', 'count']

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6493 entries, 0 to 6492
Data columns (total 2 columns):
 #   Column    Non-Null Count  Dtype
---  ------    --------------  -----
 0   datetime  6493 non-null   object
 1   count     6493 non-null   int64
dtypes: int64(1), object(1)
'''

# train을 x_train, y_train로 나누고 불필요한 columns drop
x = train.drop(['datetime', 'casual', 'registered', 'count'], axis=1)  
y = train['count']  
test.drop(['datetime'], axis=1, inplace=True)

ic(x.columns)  # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']
ic(x.shape)  # (10886, 8)   
ic(y.shape)  # (10886,)        
plt.plot(y)
plt.show() 

# 로그 변환
y = np.log1p(y)
'''
log는 태생적으로 큰 값을 작게 표기하기 위해 고안된 방법이다. 
데이터가 넓거나 한쪽으로 치우쳐진 경우에 데이터를 로그변환 해준다.
- np.log: ln(x)
- np.log1p: ln(x+1)
기본적으로 log안의 x값은 양수만 가능하다. 하지만 0에 가까운 아주 작은 양수의 경우 (ex. 0.0000000001)
음의 무한대에 가까워지게 된다. (너무 작은 값의 경우 프로그램의 계산이 -inf가 나오게됨)
이를 방지하기 위해 1을 더함으로써 0보다 큰 양수의 값을 갖게 된다.
'''
plt.plot(y)
plt.show()      

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)  


#2. 모델링    
model = Sequential()
model.add(Dense(8, input_dim=8))    
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

# EarlyStopping
es = EarlyStopping(monitor="val_loss", patience=50, mode='min', verbose=1, restore_best_weights=True)
model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=1, validation_split=0.2, callbacks=[es])

#4. 평가
loss = model.evaluate(x_test, y_test)   
print('loss값 : ', loss) 

#5. x_test값에 대한 예측 
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)  # 예측값과 실제값 비교(정확할수록 1에 가까워진다)
print("R2 : ", r2)

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

'''
y에 log 안 씌운 값
loss값 :  23817.400390625        24274.953125
R2     :  0.2464691949711344     0.24030994263465288
RMSE   :  154.32886793879092     155.80420626358412

y값 log화 시킨 후  
loss값 :   1.4510016441345215     1.432651162147522          1.4329811334609985          1.964599370956421
R2     :   0.25952479581392796    0.26834010256942986        0.2681715744937415          -0.003327715201195458
RMSE   :   1.2045752284981979     1.1969340635666597         1.1970719045094265          1.4016416861905503
'''
# epochs=5000, batch=10, patience=100  [loss] :  1.4941960573196411  [r2스코어] :  0.2741354068694233  [RMSE] :  1.222373117891978
# epochs=500, batch=1, patience=50  [loss] :  1.4933232069015503  [r2스코어] :  0.27455926401755704  [RMSE] :  1.2220161730523573


#6. 최종 테스트
results = model.predict(test)
submit_file['count'] = results 
path = "./save/bike/" 
submit_file.to_csv(path + '4thtest.csv', index=False) 
