# 다차원에서 사용할 수 있는 방법 찾아놓기

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects
import pandas as pd
aaa = np.array([[1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600,7, 800, 900, 190, 1001, 1002, 99]])
aaa = np.transpose(aaa)  # (2,13)
print(aaa.shape)         # (13,2)
# fig = plt.figure()
# fig1 = fig.add_subplot(1,2,1)
# plt.boxplot(aaa)
# plt.show()
df = pd.DataFrame(aaa, columns=['x','y'])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,
                                               [25, 50, 75])
    print('1사분위 : ', quartile_1)
    print('q2 :', q2)
    print('3사분위 :', quartile_3)
    iqr = quartile_3 - quartile_1
    print('iqr :', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

# outliers_loc = outliers(df, 'x')
print('이상치의 위치 :', outliers(df))

# # 정제 후 그래프
# Q1 = np.percentile(aaa,25)   # 4.5
# Q3 = np.percentile(aaa,75)   # 11.5
# IQR = Q3 - Q1                 # 7.0
# lower = Q1 - 1.5 * IQR        # -6.0
# upper = Q3 + 1.5 * IQR        # 22.0
# aaa = aaa[(aaa>=lower) & (aaa<=upper)]
# fig2 = fig.add_subplot(1,2,2)
# plt.boxplot(aaa)
# plt.show()


