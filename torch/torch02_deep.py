import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim     
import torch.nn.functional as F 
import time

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. data
x =  np.array([1,2,3])
y =  np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)  # unsqueeze(1): 데이터 1번째 자리의 차원을 늘려준다. -->  (3,) -> (3, 1)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)   

#2. modeling
model = nn.Sequential(
    nn.Linear(1,5),  # nn.Linear(input_dim, output_dim)
    nn.Linear(5,3),
    nn.Linear(3,4),
    nn.Linear(4,2),
    nn.Linear(2,1),
).to(DEVICE) 

#3. compile, train         
optimizer = optim.Adam(model.parameters(), lr=0.01) 
def train(model,  optimizer, x, y):
    optimizer.zero_grad()   
    hypothesis = model(x)  
    loss = F.mse_loss(hypothesis, y)  # or  loss = nn.MSELoss()(hypothesis, y)
    loss.backward()    
    optimizer.step()    
    return loss.item()  

epochs = 0
while True:
    epochs += 1
    loss = train(model, optimizer, x, y)   
    print(f'epoch : {epochs}, loss : {loss}')  # epoch : 604, loss : 0.0
    if loss == 0: break

#4. evaluate, predict
def evaluate(model, x, y):
    model.eval() 
    with torch.no_grad():  
        predict = model(x)
        loss = nn.MSELoss()(predict,y)   
    return loss.item()

loss = evaluate(model, x, y)
print(f'loss : {loss}')  # loss : 0.0

result = model(torch.Tensor([[4]]).to(DEVICE))  
print(f'4의 예측값 : {result.item()}')  # 4의 예측값 : 4.000000476837158