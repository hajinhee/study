import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. data
x_train = np.array([1,2,3,4,5,6,7])  # (7,)
x_test = np.array([8,9,10])  # (3,)
y_train = np.array([1,2,3,4,5,6,7])  # (7,)
y_test = np.array([8,9,10])  # (3,)

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)  # (7, 1)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)  # (3, 1)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)  # (7, 1)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)  # (3, 1)

#2. modeling
model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.ReLU(),
    nn.Linear(3,4),
    nn.Linear(4,1),       
).to(DEVICE)

#3. compile, train
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1)
def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hyporthesis = model(x)
    loss = criterion(hyporthesis, y)
    loss.backward() 
    optimizer.step()
    return loss.item()

EPOCHS = 100
for epoch in range(1,EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch : ', epoch, 'loss : ', loss)  # epoch :  100 loss :  0.0006561267655342817
    
# evaluate, predict
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        predict = model(x)
        loss = criterion(predict, y)
        return loss.item() 

loss = evaluate(model, criterion, x_test, y_test)
print('loss: ', loss)  # loss:  0.0030460411217063665

result = model(x_test.to(DEVICE))
print(result.cpu().detach().numpy()) 
'''
[[7.9529424]
 [8.945175 ]
 [9.937407 ]]
'''

result = model(torch.Tensor(([[11]])).to(DEVICE)).to(DEVICE)
print(result.cpu().detach().numpy())  # [[10.936045]]

'''
cpu(): GPU 메모리에 올려져 있는 tensor를 cpu 메모리로 복사
detach(): graph(tensor에서 이루어진 모든 연산을 추적해서 기록한 것)에서 분리한 새로운 tensor를 반환
numpy(): tensor를 numpy로 변환
'''