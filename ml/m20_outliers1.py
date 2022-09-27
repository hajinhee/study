import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


aaa = np.array([1,2, -70, 4, 5, 6, 7, 8, 50, 60, 70, 12, 13])
fig = plt.figure()
fig1 = fig.add_subplot(1,2,1)

plt.boxplot(aaa)
plt.show()

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

outliers_loc = outliers(aaa)
print('이상치의 위치 :', outliers(aaa))

# percentile  분위수 꺼내기 / [25, 50, 75] : 1사분위, 중위값, 3사분위

'''
1사분위 :  3.0
q2 : 6.0
3사분위 : 49.0
(array([ 2, 10], dtype=int64),)

return np.where((data_out>upper_bound) | (data_out<lower_bound)) >> 이상치를 추출한다. 
이상치의 위치 : (array([ 2,  8,  9, 10], dtype=int64),) >> index 위치
 | : 또는 
'''

# outlier 제거
Q1 = np.percentile(aaa,25)   # 4.5
Q3 = np.percentile(aaa,75)   # 11.5
IQR = Q3 - Q1                 # 7.0

lower = Q1 - 1.5 * IQR        # -6.0
upper = Q3 + 1.5 * IQR        # 22.0

aaa = aaa[(aaa>=lower) & (aaa<=upper)]

fig2 = fig.add_subplot(1,2,2)
plt.boxplot(aaa)
plt.show()


