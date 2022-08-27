# import numpy as np

f = lambda x: x**2 - 4*x + 6

def f2(x):
    temp = x**2 - 4*x + 6
    return temp

gradient = lambda x: 2*x -4     # gradient는 f에 대한 기울기.weight.이다

def gradient2(x):
    temp = 2*x-4
    return temp

# 둘은 같다.

# 미분 -> 2차 3차함수에서 각 지점에 대한 기울기.

# 초기세팅값
x = -10           
epochs = 25
learning_rate = 0.25

print("step\t x\t\t f(x)")
print("{:02d}\t {:6.5f}\t {:6.5f}\t".format(0,x,f(x)))      # 초기값 출력

for i in range(epochs):
    x = x - learning_rate * gradient(x)
    
    print("{:02d}\t  {:6.5f}\t {:1.5f}\t".format(i+1,x,f(x)))