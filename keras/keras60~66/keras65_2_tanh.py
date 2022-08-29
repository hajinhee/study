import numpy as np, matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1)
y = np.tanh(x)  # 내장함수가 들어있다. sigmoid: 0.0~1.0 tanh는 -1.0~1.0

plt.plot(x, y)
plt.grid()
plt.show()