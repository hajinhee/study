import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim     # optimizer
import torch.nn.functional as F # Loss같은거 기능들? 들어가 있다.
import time

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. data
x =  np.array([1,2,3])
y =  np.array([1,2,3])
x = torch.FloatTensor(x)  # tensor([1., 2., 3.])
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)   
# scaling
x = (x - torch.mean(x)) / torch.std(x)  # 평균을 빼고 표준편차로 나눈다(Standard) -> tensor([-1.,  0.,  1.])
x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)

#2. modeling
model = nn.Linear(1, 1).to(DEVICE)   

#3. compile, train      
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01) 
def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()  # 기울기 초기화
    hypothesis = model(x)  # 가설 정의
    loss = F.mse_loss(hypothesis, y)  # or  loss = criterion(hypothesis, y) 
    loss.backward()    
    optimizer.step()    
    return loss.item()  

epochs = 0
while True:
    epochs += 1
    loss = train(model, criterion, optimizer, x, y) 
    print(f'epoch : {epochs}, loss : {loss}')  # epoch : 8015, loss : 0.0
    if loss == 0: break

#4. evaluate, predict
def evaluate(model, criterion, x, y):
    model.eval()       
    with torch.no_grad():   
        y_pred = model(x)
        loss = criterion(y_pred, y)
    return loss.item()

loss = evaluate(model, criterion, x, y)
print(f'loss : {loss}')  # loss : 0.0

result = model(torch.Tensor([[4]]).to(DEVICE)) 
print(f'4의 예측값 : {result.item()}')  # 4의 예측값 : 6.0