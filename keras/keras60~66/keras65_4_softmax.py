import numpy as np, matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(-5, 5, 0.1)
y = softmax(x)      

ratio = y
labels = y

plt.pie(ratio, labels, shadow=True, startangle=90)
# plt.plot(x,y)
# plt.grid()
plt.show()

# activation의 주 목적.     레이어에서 다음레이어로 건너갈때 값을 제한시켜줌으로써
# 특징을 유지한채로 값만 줄여서 연산량을 줄여준다.