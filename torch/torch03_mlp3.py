import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim     
import torch.nn.functional as F 
import time

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. data
x = np.array([range(10), range(21,31), range(201,211)])  # (3, 10)
y = np.array([[1,2,3,4,5,6,7,8,9,10], [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3], [10,9,8,7,6,5,4,3,2,1]])  # (3, 10)
x = np.transpose(x)  # (10, 3)
y = np.transpose(y)  # (10, 3)
x = torch.FloatTensor(x).to(DEVICE)  # torch.Size([10, 3])
y = torch.FloatTensor(y).to(DEVICE)  # torch.Size([10, 3])

#2. modeling
model = nn.Sequential(
    nn.Linear(3,5),
    nn.Linear(5,3),
    nn.Linear(3,4),
    nn.Linear(4,2),
    nn.Linear(2,3),
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
    loss = train(model, criterion, optimizer, x, y)   
    if epochs % 100 == 0 : print(f'epoch : {epochs}, loss : {loss}')  # epoch : 500, loss : 0.024356674402952194
    if loss < 0.006: break

#4. evaluate, predict
def evaluate(model,criterion, x, y):
    model.eval()    
    with torch.no_grad():  
        predict = model(x)
        loss = criterion(predict, y)
    return loss.item()

loss = evaluate(model, criterion, x, y)
print(f'loss : {loss}')  # loss : 0.005965216085314751

result = model(torch.Tensor([[9,30,210]]).to(DEVICE))  
print(f'[9,30,210]의 예측값 : {result}')  # [9,30,210]의 예측값 : tensor([[10.0021,  1.4554,  1.0002]]
