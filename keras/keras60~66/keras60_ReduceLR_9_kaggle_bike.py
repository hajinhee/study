from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense
import numpy as np,pandas as pd,time
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from icecream import ic

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

#1. loda data
path = 'keras/data/kaggle/bike/'
train = pd.read_csv(path + 'train.csv')    
test_file = pd.read_csv(path + 'test.csv')                   
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

ic(train.columns)
'''
['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
'''
ic(train.head(20))
'''
                datetime  season  holiday  workingday  weather   temp   atemp  humidity  windspeed  casual  registered  count
0   2011-01-01 00:00:00       1        0           0        1   9.84  14.395        81     0.0000       3          13     16      
1   2011-01-01 01:00:00       1        0           0        1   9.02  13.635        80     0.0000       8          32     40
2   2011-01-01 02:00:00       1        0           0        1   9.02  13.635        80     0.0000       5          27     32      
3   2011-01-01 03:00:00       1        0           0        1   9.84  14.395        75     0.0000       3          10     13      
4   2011-01-01 04:00:00       1        0           0        1   9.84  14.395        75     0.0000       0           1      1
5   2011-01-01 05:00:00       1        0           0        2   9.84  12.880        75     6.0032       0           1      1      
'''
ic(train.corr())
'''
            season   holiday  workingday   weather      temp     atemp  humidity  windspeed    casual  registered     count
season      1.000000  0.029368   -0.008126  0.008879  0.258689  0.264744  0.190610  -0.147121  0.096758    0.164011  0.163439       
holiday     0.029368  1.000000   -0.250491 -0.007074  0.000295 -0.005215  0.001929   0.008409  0.043799   -0.020956 -0.005393       
workingday -0.008126 -0.250491    1.000000  0.033772  0.029966  0.024660 -0.010880   0.013373 -0.319111    0.119460  0.011594       
weather     0.008879 -0.007074    0.033772  1.000000 -0.055035 -0.055376  0.406244   0.007261 -0.135918   -0.109340 -0.128655
temp        0.258689  0.000295    0.029966 -0.055035  1.000000  0.984948 -0.064949  -0.017852  0.467097    0.318571  0.394454       
atemp       0.264744 -0.005215    0.024660 -0.055376  0.984948  1.000000 -0.043536  -0.057473  0.462067    0.314635  0.389784       
humidity    0.190610  0.001929   -0.010880  0.406244 -0.064949 -0.043536  1.000000  -0.318607 -0.348187   -0.265458 -0.317371
windspeed  -0.147121  0.008409    0.013373  0.007261 -0.017852 -0.057473 -0.318607   1.000000  0.092276    0.091052  0.101369       
casual      0.096758  0.043799   -0.319111 -0.135918  0.467097  0.462067 -0.348187   0.092276  1.000000    0.497250  0.690414       
registered  0.164011 -0.020956    0.119460 -0.109340  0.318571  0.314635 -0.265458   0.091052  0.497250    1.000000  0.970948       
count       0.163439 -0.005393    0.011594 -0.128655  0.394454  0.389784 -0.317371   0.101369  0.690414    0.970948  1.000000

'''
# drop columns
x = train.drop(['casual', 'registered', 'count'], axis=1) 

# split date
x['datetime'] = pd.to_datetime(x['datetime'])
x['year'] = x['datetime'].dt.year
x['month'] = x['datetime'].dt.month
x['day'] = x['datetime'].dt.day
x['hour'] = x['datetime'].dt.hour

# split x, y data
x = x.drop('datetime', axis=1)
y = train['count']  

# check target
ic(np.unique(y, return_counts=True))
ic(y.head(10))
'''
0    16
1    40
2    32
3    13
4     1
5     1
6     2
7     3
8     8
9    14
'''
y = np.log1p(y)
ic(y.head(10))
'''
0    2.833213
1    3.713572
2    3.496508
3    2.639057
4    0.693147
5    0.693147
6    1.098612
7    1.386294
8    2.197225
9    2.708050
'''

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=49)  

# scaling
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. modeling
model = Sequential()
model.add(Dense(16, input_dim=12))    
model.add(Dense(24)) 
model.add(Dense(32)) 
model.add(Dense(24)) 
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

#3. compile, train
learning_rate = 0.001           
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mse', optimizer=optimizer, metrics='mae')
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto', verbose=1, min_lr=0.0001, factor=0.5)   

start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=5, validation_split=0.2, callbacks=[reduce_lr, es])
end = time.time()-start

#4. evaluate, predict
loss, mae = model.evaluate(x_test, y_test, batch_size=32)
print(loss, mae)

y_predict = model.predict(x_test)
r2 = r2_score(np.round(np.expm1(y_test)), np.round(np.expm1(y_predict)))  #  np.exp2는 밑이 2인 경우
rmse = RMSE(np.round(np.expm1(y_test)), np.round(np.expm1(y_predict)))
ic(r2, rmse)

'''
r2: 0.21269721912115502, rmse: 163.12456298383913
'''