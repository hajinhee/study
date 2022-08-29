import numpy as np, matplotlib.pyplot as plt

def softmax(x):  # 0.0~1.0 
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(-5, 5, 0.1)
y = softmax(x)      

ratio = y
labels = y

plt.pie(ratio, labels, shadow=True, startangle=90)
# plt.plot(x,y)
# plt.grid()
plt.show()

# activation funtion: 레이어에서 다음 레이어로 넘어갈 때 특징은 유지한채로 값만 제한함으로써 연산량을 줄여준다.