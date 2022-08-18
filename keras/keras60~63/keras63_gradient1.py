import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 - 4*x + 6

x = np.linspace(-1, 6, 100) # -1부터 6까지 100개로 조각을 내었다.

y = f(x)

########## 그려보자
plt.plot(x,y,'k-')
plt.plot(2,2,'sk')  # 2,2지점에 검정 사각형이 생겼다.
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()