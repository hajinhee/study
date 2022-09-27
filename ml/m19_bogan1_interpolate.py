# 결측치 처리
# 1. 행 또는 열 삭제
# 2. 임의의 값 
    # FillNa - 0, ffill, bfill, 중위값, 평균값... 76767
# 3. 보간 - interpolate
# 4. 모델링 - predict
# 5. 부스팅 계열 - 통상 결측치, 이상치에 대해 자유롭다. 

# dataframe : 행렬
# Series : 벡터

import pandas as pd
from datetime import datetime
import numpy as np

dates = ['1/24/2022','1/25/2022','1/26/2022',
         '1/27/2022','1/28/2022']
dates = pd.to_datetime(dates)
print(dates)
ts = pd.Series([2, np.nan, np.nan, 8, 10], index = dates)
print(ts)

ts = ts.interpolate()
print(ts)


'''
2022-01-24     2.0
2022-01-25     NaN
2022-01-26     NaN
2022-01-27     8.0
2022-01-28    10.0
dtype: float64
2022-01-24     2.0
2022-01-25     4.0
2022-01-26     6.0
2022-01-27     8.0
2022-01-28    10.0
dtype: float64

가장 단순한 선형 회귀 사용 y = wx + b
'''