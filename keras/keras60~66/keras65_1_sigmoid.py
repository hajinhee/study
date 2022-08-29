# sigmoid 코드 구현

import numpy as np, matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))   # np.exp(): 자연상수 e를 밑으로 하는 로그함수이다. 자연상수 e는 2.71828182846... 의 값을 가지는 무리수

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)  # 어떤 x값을 넣더라도 결과값이 0과 1사이로 수렴한다.

plt.plot(x, y)
plt.grid()
plt.show()