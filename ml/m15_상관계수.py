import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

#1. 데이터
datasets = load_iris()
print(datasets.feature_names) # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']-;
x = datasets.data
y = datasets.target

df = pd.DataFrame(x, columns = datasets['feature_names'])
# df = pd.DataFrame(x, columns = [['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])
# print(df)
df['Target(y)'] = y
print(df)

print('==================================상관계수 히트 맵================================================')
print(df.corr())
import matplotlib.pyplot as plt
import seaborn as sns  # 맷플로립보다 더 이쁘게 나옴
plt.figure(figsize=(10,10))
sns.set(font_scale = 1)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar = True)
plt.show()
## y와 높은 상관관계를 갖고 있는 x의 피쳐는 
# y =wx+b로 빠르게 연산해서 찾은 상관 관계도이다.
