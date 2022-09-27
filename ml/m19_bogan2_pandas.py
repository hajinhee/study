import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, np.nan, 8, 10],
                     [2, 4, np.nan, 8, np.nan],
                     [np.nan, 4, np.nan, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]])
# print(data.shape)  # (4, 5)
data = data.transpose() 
# print(data.shape)
data.columns = ['a', 'b', 'c', 'd']
# print(data)

# 결측치 확인
print(data.isnull())
# print(data.isnull().sum())
print(data.info())
'''
      a    b     c    d
0   2.0  2.0   NaN  NaN
1   NaN  4.0   4.0  4.0
2   NaN  NaN   NaN  NaN
3   8.0  8.0   8.0  8.0
4  10.0  NaN  10.0  NaN
'''
# 1. 결측치 삭제
# print(data.dropna())
# print(data.dropna(axis=0))
# print(data.dropna(axis=1))
'''
print(data.dropna())
     a    b    c    d
3  8.0  8.0  8.0  8.0
print(data.dropna(axis=0))
     a    b    c    d
3  8.0  8.0  8.0  8.0
print(data.dropna(axis=1))
Empty DataFrame
Columns: []
Index: [0, 1, 2, 3, 4]
# '''
#2-1. 특정값 - 평균
# means = data.mean() 
# print(means)
# data1 = data.fillna(means)
# print(data1)
# '''
#       a    b     c    d
# 0   2.0  2.0   NaN  NaN
# 1   NaN  4.0   4.0  4.0
# 2   NaN  NaN   NaN  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN
# a    6.666667
# b    4.666667
# c    7.333333
# d    6.000000
# dtype: float64

# - data = data.fillna(means)
#            a         b          c    d
# 0   2.000000  2.000000   7.333333  6.0
# 1   6.666667  4.000000   4.000000  4.0
# 2   6.666667  4.666667   7.333333  6.0
# 3   8.000000  8.000000   8.000000  8.0
# 4  10.000000  4.666667  10.000000  6.0
# '''
# #2-2. 특정값 - 중위값(중간 위치)
# meds = data.median()
# print(meds)
# data2 = data.fillna(meds)
# print(data2)
# # '''
# # a    8.0
# # b    4.0
# # c    8.0
# # d    6.0
# #       a    b     c    d
# # 0   2.0  2.0   8.0  6.0
# # 1   8.0  4.0   4.0  4.0
# # 2   8.0  4.0   8.0  6.0
# # 3   8.0  8.0   8.0  8.0
# # 4  10.0  4.0  10.0  6.0
# # '''

# #2-3. 특정값 - ffill, bfill
# data2 = data.fillna(method='ffill', limit=1)
# print(data2)
# data2 = data.fillna(method='bfill', limit=1)
# print(data2)

# #2-4. 특정값 - 채우기
# data2 = data.fillna(0)
# print(data2)

# ##################################### 특정 컬럼만!! ################################################

means = data['a'].mean()
print(means)  # 6.666666666666667
data['a'] = data['a'].fillna(means)
print(data)

median = data['b'].median()
print(median)
data['b'] = data['b'].fillna(median)
print(data)






