from sklearn.datasets import fetch_covtype             
from tensorflow.keras.models import Sequential         
from tensorflow.keras.layers import Dense               
import numpy as np                                      
from sklearn.model_selection import train_test_split    
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical 
from sklearn.preprocessing import OneHotEncoder            
import pandas as pd
from sklearn.datasets import load_wine

datasets = load_wine()
x = datasets.data  # (178, 13)  
y = datasets.target  # (178,)
print(np.unique(y))  # [0 1 2]


'''
원핫인코딩을 하는 3가지 방법
'''

# 1. pandas의 get_dummies
y = pd.get_dummies(y)
print(y)  # (178, 3)
'''
     0  1  2
0    1  0  0
1    1  0  0
2    1  0  0
3    1  0  0
4    1  0  0
..  .. .. ..
173  0  0  1
174  0  0  1
175  0  0  1
176  0  0  1
177  0  0  1

[178 rows x 3 columns]
'''


# 2. tensorflow의 to_categorical
y = to_categorical(y)
print(y)  # (178, 3)
'''
[[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
'''

# 3. sklearn의 OneHotEncoder 
en = OneHotEncoder(sparse=False)  # sparse=True가 디폴트이며 이는 Matrix를 반환한다. 원핫인코딩에서 필요한 것은 array이므로 sparse 옵션에 False를 넣어준다.       
y = en.fit_transform(y.reshape(-1, 1))  # 2차원변환 해주기 위해 행의 자리에 -1넣고 열이 1개라서 1넣은 것. 그러면 세로 배열된다. 가로 배열은(1,-1)이다.
print(y)  # (178, 3)  
'''
[[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
'''

