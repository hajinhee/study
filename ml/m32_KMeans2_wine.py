from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

datasets = load_wine()
irisDF = pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(irisDF)     # <class 'numpy.ndarray'>

kmeans = KMeans(n_clusters=3, random_state=66)  # max_iter : 최대 몇 번 작업할 것인가?
kmeans.fit(irisDF)

irisDF['cluster'] = kmeans.labels_
irisDF['target'] = datasets.target

score = accuracy_score(irisDF['target'],irisDF['cluster'])
print(score)