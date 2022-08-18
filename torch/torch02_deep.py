import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim     
import torch.nn.functional as F 
import time

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. 데이터 정제해서 값 도출
x =  np.array([1,2,3])
y =  np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)   
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)   

#2. 모델구성
# model = nn.Linear(1, 1).to(DEVICE)     
model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,3),
    nn.Linear(3,4),
    nn.Linear(4,2),
    nn.Linear(2,1),
).to(DEVICE)    #Linear을 Sequential로 묶었다.

#3. 컴파일, 훈련         
optimizer = optim.Adam(model.parameters(), lr=0.01) 

def train(model,  optimizer, x, y):
    #model.train()   # 훈련모드 default로 들어가 있음.
    optimizer.zero_grad()   
    
    hypothesis = model(x)  
    
    # loss = nn.MSELoss()(hypothesis,y)   # 이렇게 선언하면 돌아간다.
    loss = F.mse_loss(hypothesis,y)     # 이것도 잘 돌아간다.
    loss.backward()    
    optimizer.step()    
    return loss.item()  

epochs = 0
while True:
    epochs += 1
    loss = train(model,optimizer,x,y)    # 현재는 data loader 쓰지 않는 상태
    print(f'epoch : {epochs}, loss : {loss}')
    # time.sleep(0.1)
    if loss == 0: break
print('==================================================================================')

#4. 평가, 예측
def evaluate(model, x, y):
    model.eval()        # torch는 eval로 평가한다. 평가모드

    with torch.no_grad():   # gradient를 갱신하지않겠다.
        predict = model(x)
        loss2 = nn.MSELoss()(predict,y)   

    return loss2.item()

loss2 = evaluate(model, x, y)
print(f'최종 loss : {loss2}')

result = model(torch.Tensor([[4]]).to(DEVICE))  
print(f'4의 예측값 : {result.item()}')