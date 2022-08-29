import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

f = lambda x: x**2 - 4*x + 6

x = np.linspace(-1, 6, 100) # -1부터 6까지 100개로 조각을 내었다.
ic(x)

y = f(x)
ic(y)

########## visualization ##########
plt.plot(x, y, 'k-')
plt.plot(2, 2, 'sk')  
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()