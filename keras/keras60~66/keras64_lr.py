import math

x = 10
y = 10  # target
w = 0.5  # weight
lr = 0.0001
epochs = 100

# 행렬연산에서는 wx xw의 앞뒤 순서 바뀌면 차이가 크다.
# tensor1이나 파이토치에서는 x에 weight를 곱한다.

for i in range(epochs):
    predict = x * w
    loss = math.sqrt((predict-y)**2)   # 이 predict값을 y값과 같아질때까지 계속 값을 갱신해간다.  
    # 이 때 갱신해가는 방법이 weight의 갱신인데 weight의 갱신을 해주는 변수가 learning_rate이다.
        
    up_predict = x * (w + lr)
    up_loss = (y - up_predict)**2
    
    down_predict = x * (w - lr)
    down_loss = (y - down_predict)**2
    
    if up_loss > down_loss:
        w = w - lr
    else:
        w = w + lr 
        
    print(f'Step{i+1}\t Loss(MAE) : {round(loss,4)}\t Predict : {round(predict,4)}\t  Weight : {w}')
    
'''
Step1    Loss(MAE) : 5.0         Predict : 5.0    Weight : 0.5001
Step2    Loss(MAE) : 4.999       Predict : 5.001          Weight : 0.5002
Step3    Loss(MAE) : 4.998       Predict : 5.002          Weight : 0.5003
Step4    Loss(MAE) : 4.997       Predict : 5.003          Weight : 0.5004
Step5    Loss(MAE) : 4.996       Predict : 5.004          Weight : 0.5005
Step6    Loss(MAE) : 4.995       Predict : 5.005          Weight : 0.5005999999999999
Step7    Loss(MAE) : 4.994       Predict : 5.006          Weight : 0.5006999999999999
Step8    Loss(MAE) : 4.993       Predict : 5.007          Weight : 0.5007999999999999
Step9    Loss(MAE) : 4.992       Predict : 5.008          Weight : 0.5008999999999999
Step10   Loss(MAE) : 4.991       Predict : 5.009          Weight : 0.5009999999999999
Step11   Loss(MAE) : 4.99        Predict : 5.01   Weight : 0.5010999999999999
'''