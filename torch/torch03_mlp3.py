import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim     
import torch.nn.functional as F 
import time

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. 데이터 정제해서 값 도출
x = np.array([range(10), range(21,31), range(201,211)])
y = np.array([[1,2,3,4,5,6,7,8,9,10],[1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3],[10,9,8,7,6,5,4,3,2,1]]) 

x = np.transpose(x) # (10, 3)
y = np.transpose(y) # (10, 3)

x = torch.FloatTensor(x).to(DEVICE)   
y = torch.FloatTensor(y).to(DEVICE)  

# print(x.shape,y.shape)  # torch.Size([10, 3]) torch.Size([10, 1])

#2. 모델구성
model = nn.Sequential(
    nn.Linear(3,5),
    nn.Linear(5,3),
    nn.Linear(3,4),
    nn.Linear(4,2),
    nn.Linear(2,3),
).to(DEVICE)    #Linear을 Sequential로 묶었다.

#3. 컴파일, 훈련     
criterion = nn.MSELoss()    
optimizer = optim.Adam(model.parameters(), lr=0.01) 

def train(model, criterion, optimizer, x, y):
    #model.train()   # 훈련모드 default로 들어가 있음.
    optimizer.zero_grad()   
    
    hypothesis = model(x)  
    
    loss = criterion(hypothesis, y)
    # loss = nn.MSELoss()(hypothesis,y)   # 이렇게 선언하면 돌아간다.
    # loss = F.mse_loss(hypothesis,y)     # 이것도 잘 돌아간다.
    loss.backward()    
    optimizer.step()    
    return loss.item()  

epochs = 0
while True:
    epochs += 1
    loss = train(model,criterion,optimizer,x,y)    # 현재는 data loader 쓰지 않는 상태
    if epochs % 100 == 0 : print(f'epoch : {epochs}, loss : {loss}')
    # time.sleep(0.1)
    if loss < 0.006: break
print('==================================================================================')

#4. 평가, 예측
def evaluate(model,criterion, x, y):
    model.eval()        # torch는 eval로 평가한다. 평가모드

    with torch.no_grad():   # gradient를 갱신하지않겠다.
        predict = model(x)
        loss2 = criterion(predict, y)
        # loss2 = nn.MSELoss()(predict,y)   

    return loss2.item()

loss2 = evaluate(model, criterion,x, y)
print(f'최종 loss : {loss2}')

result = model(torch.Tensor([[9,30,210]]).to(DEVICE))  
print(f'[9,30,210]의 예측값 : {result}')
