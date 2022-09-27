import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [np.nan, 4, np.nan, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]])
print(data.shape)  # (4, 5)
data = data.transpose()
# data = data.reshape(1,-1)
print(data.shape)
data.columns = ['a', 'b', 'c', 'd']
print(data)

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.impute import SimpleImputer
# pandas에서 했던 것을 한번에 할 수 있다. 
# imputer = SimpleImputer(strategy = 'mean')
# imputer = SimpleImputer(strategy = 'median')
# imputer = SimpleImputer(strategy = 'most_frequent')  # most_frequent : 가장 빈번하게 사용한 값
# imputer = SimpleImputer(strategy = 'constant') # 0으로 입력됨
imputer = SimpleImputer(strategy = 'median')

imputer.fit(data[['a']])
data2= imputer.transform(data[['a']])
print(data2)

# fit에는 dataframe이 들어가는데, 우리는 컬럼만 바꾸고 싶다. 
# 시리즈를 넣으면 에러가 난다. 