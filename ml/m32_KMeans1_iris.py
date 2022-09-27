from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

datasets = load_iris()
irisDF = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(irisDF)     # <class 'numpy.ndarray'>

kmeans = KMeans(n_clusters=3, random_state=66)  # max_iter : 최대 몇 번 작업할 것인가?
kmeans.fit(irisDF)

irisDF['cluster'] = kmeans.labels_
irisDF['target'] = datasets.target

score = accuracy_score(irisDF['target'],irisDF['cluster'])
print(score)

'''
kmaen
y 값 자체를 지정해 주지 않아도 된다. 
n_clusters = 타겟 갯수와 동일하게 
'''

from sklearn.datasets import load_iris  
from sklearn.cluster import KMeans # y가 없는 거 비지도 학습
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


datasets=load_iris()

# x = datasets.data
# y = datasets.target

x=pd.DataFrame(datasets.data, columns=[datasets.feature_names])

#print(type(x))
print(x)

kmeans = KMeans(n_clusters=3)  #분류에서만 가능
kmeans.fit(x)

y_predict = kmeans.predict(x)
print(kmeans.labels_)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 2 2 1 2 2 2 2
#  2 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 1 2 2 2 1 2
#  2 1]

print(datasets.target)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]
'''
n_clusters : int, optional, default: 8
  k개(분류할 수) = 클러스터의 중심(centeroid) 수

init : {‘k-means++’, ‘random’ or an ndarray}
  Method for initialization, defaults to ‘k-means++’:
  ‘k-means++’
  ‘random’
  np.array([[1,4],[10,5],[16,2]]))  

n_init : int, default: 10
  큰 반복 수 제한 (클러스터의 중심(centeroid) 위치)

max_iter : int, default: 300
  작은 반복수 제한

tol : float, default: 1e-4
  inertia (Sum of squared distances of samples to their closest cluster center.) 가 tol 만큼 줄어 들지 않으면 종료 (조기 종료)
'''

x['cluster'] = kmeans.labels_
x['target'] = datasets.target

# iris_result = x.groupby(['target', 'cluster']).count()
# print(iris_result)

print(accuracy_score(datasets.target,kmeans.labels_))


from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import completeness_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import v_measure_score


print(silhouette_score(x['cluster'], x['target'])) #0.6322939531368102 #실루엣 계수: 군집간 거리는 멀고 군집내 거리는 가까울수록 점수 높음 (0~1), 0.5 보다 크면 클러스터링이 잘 된거라 평가
print(adjusted_mutual_info_score(datasets.target, y_predict)) #1.0
print(adjusted_rand_score(datasets.target, y_predict)) #1.0
print(completeness_score(datasets.target, y_predict)) #1.0
print(fowlkes_mallows_score(datasets.target, y_predict)) #1.0
print(homogeneity_score(datasets.target, y_predict)) #1.0
print(mutual_info_score(datasets.target, y_predict)) #1.077556327066801
print(normalized_mutual_info_score(datasets.target, y_predict)) #1.0
print(v_measure_score(datasets.target, y_predict)) #1.0