import numpy as np, matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def leaky_lelu(a, x):
    return np.maximum(a*x, x)
# 하이퍼파라미터 α가 이 함수가 새는(leaky) 정도를 결정한다. 새는 정도란 x<0일 때 이 함수의 기울기이며, 일반적으로 0.01로 설정

def elu(a, x):
    return (x>0)*x + (x<=0)*(a*(np.exp(x)-1))
    
def selu(alpha, scale, x):
    return np.where(x<=0, scale*alpha*(np.exp(x)-1), scale*x)

x = np.arange(-5, 5, 0.1)
y = selu(0.5, 0.5, x)

plt.plot(x, y)
plt.grid()
plt.show()
