import numpy as np
import pandas as pd

aaa = np.array([[1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600,7, 800, 900, 190, 1001, 1002, 99]])
aaa = np.transpose(aaa)

# df = pd.DataFrame(aaa, columns=['x','y'])

# data1 = df[['x']]
# data2 = df[['y']]

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.15)
pred = outliers.fit_predict(aaa)
print(pred) # (13,)

# b = list(pred)
# print(b.count(-1))
# index_for_outlier = np.where(pred == -1)
# print('outier indexex are', index_for_outlier)
# outlier_value = aaa[index_for_outlier]
# print('outlier_value :', outlier_value)
# outliers.fit(data1)
# outliers.fit(data2)
# result1 = outliers.predict(data1)
# result2 = outliers.predict(data2)

# print(result1)
# print(result2)
'''
# 1개만 할 경우
data2[['a']] = imputer.fit_transform(data[['a']])

# 2개이상 할 경우
data2[['a','c']] = imputer.fit_transform(data[['a','c']])
data2[['b','d']] = imputer.fit_transform(data[['b','d']])
'''
'''
오염도 %에 따라 출력함 
[ 1  1  1  1  1  1  1  1  1 -1 -1  1  1]  : -1이 outlier 자리이다.
[ 1  1 -1  1  1  1  1  1  1 -1 -1  1  1]
[ 1  1 -1  1  1  1  1  1 -1 -1 -1  1  1]
ValueError: contamination must be in (0, 0.5], got: 0.600000
'''

'''

import numpy as np

aaa = np.array( [
                [1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99]
                ] ) # 2행 13열

# (2, 13) => (13, 2)
aaa = np.transpose(aaa)

print(aaa)
def outliers(data_out):
    try:
        i=0
        while True:
            # 반복설정
            i=i+1

            if data_out[:,i-1:i] is not None :
               quantile_1, q2, quantile_3 = np.percentile(data_out[:,i-1:i], [25,50,75])
               print(i,"행")
               print("1사분위 : ", quantile_1)
               print("q2 : ", q2)
               print("3사분위 : ", quantile_3)
               
               iqr = quantile_3 - quantile_1
               print("iqr : ", iqr)
               print("\n")
               lower_bound = quantile_1 - (iqr * 1.5)
               upper_bound = quantile_3 + (iqr * 1.5)

            else:
                return np.where((data_out[:,i-1:i] > upper_bound) |        #  이 줄 또는( | )
                            (data_out[:,i-1:i] < lower_bound))         #  아랫줄일 경우 반환
    except Exception:
        pass

print( outliers(aaa) )

'''