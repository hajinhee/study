import math

x = 10
y = 10                      # 목표값
w = 0.5                     # 가중치 초기값
lr = 0.0001                   
epochs = 8000

# 행렬연산에서는 wx xw의 앞뒤 순서 바뀌면 차이가 크다.
# tensor1이나 파이토치에서는 x에 weight를 곱한다.

for i in range(epochs):
    
    predict = x * w
    loss = math.sqrt((predict -y)**2)   # MAE
        
    up_predict = x * (w + lr)
    up_loss = (y - up_predict)**2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict)**2
    
    if up_loss > down_loss:
        w = w - lr
    else:
        w = w + lr 
        
    print(f"Step{i+1}\t Loss(MAE) : {round(loss,4)}\t Predict : {round(predict,4)}\t  Weight : {w}")
    

# 이 반복문은 파이토치, tensor1,2에서 경사하강법에 의하여 모델이 최적의 값을 찾아가는 과정을 그대로 표현하고 있다.
# 초기값 x와 y각 각각 주어지고 (여기서는 1개)   모델은 weight와 learning_rate로 각epoch마다 predict값을 구하면서
# 이 predict값을 y값과 같아질때까지 계속 값을 갱신해간다. 
# 이 때 갱신해가는 방법이 weight의 갱신인데 weight의 갱신을 해주는 변수가 learning_rate이다. 

