import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim     
import torch.nn.functional as F 
import time

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. data
x = np.array([[1,2,3,4,5,6,7,8,9,10], [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]])  # (2, 10)
y = np.array([11,12,13,14,15,16,17,18,19,20])  # (10,)

x = np.transpose(x) # (2, 10) -> (10, 2)
x = torch.FloatTensor(x).to(DEVICE)  # torch.Size([10, 2])
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)  # (10,) -> (10, 1)  torch.Size([10, 1])

#2. modeling
model = nn.Sequential(
    nn.Linear(2,5),
    nn.Linear(5,3),
    nn.Linear(3,4),
    nn.Linear(4,2),
    nn.Linear(2,1),
).to(DEVICE)  

#3. compile, train     
criterion = nn.MSELoss()    
optimizer = optim.Adam(model.parameters(), lr=0.01) 

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()   
    hypothesis = model(x)  
    loss = criterion(hypothesis, y)  # or  loss = nn.MSELoss()(hypothesis, y)  # loss = F.mse_loss(hypothesis, y)
    loss.backward()    
    optimizer.step()    
    return loss.item()  

epochs = 0
while True:
    epochs += 1
    loss = train(model,criterion,optimizer, x, y) 
    print(f'epoch : {epochs}, loss : {loss}')  # epoch : 3468, loss : 9.910431799653452e-07
    # time.sleep(0.1)
    if loss < 0.000001: break

#4. evaluate, predict
def evaluate(model,criterion, x, y):
    model.eval()    
    with torch.no_grad(): 
        predict = model(x)
        loss = criterion(predict, y)
    return loss.item()

loss = evaluate(model, criterion,x, y)
print(f'loss : {loss}')  # loss : 9.811618610910955e-07

result = model(torch.Tensor([[10,1.3]]).to(DEVICE))  
print(f'[10,1.3]의 예측값 : {result.item()}')  # [10,1.3]의 예측값 : 19.998228073120117
