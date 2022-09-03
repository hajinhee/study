import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim  # optimizer
import torch.nn.functional as F 
import time
from icecream import ic 

#1. data
x =  np.array([1,2,3])  # (3,)
y =  np.array([1,2,3])  # (3,)
x = torch.FloatTensor(x).unsqueeze(1) 
y = torch.FloatTensor(y).unsqueeze(1)  
''' 
FloatTensor(): 구체적인 텐서타입을 정의하여 텐서로 변환한다.
unsqueeze(1): 데이터 1번째 자리의 차원을 늘려준다. -->  (3,) -> (3, 1)
'''

ic(x.shape, y.shape)  # torch.Size([3, 1]), torch.Size([3, 1])
ic(x, y)
'''
x: tensor([[1.],
           [2.],
           [3.]])
y: tensor([[1.],
           [2.],
           [3.]])
''' 

#2. modeling
model = nn.Linear(1, 1)  # torch는 input이 앞에 위치한다. --> nn.Linear(input_dim, output_dim)

#3. compile, train      
criterion = nn.MSELoss()  # MSELoss 정의
optimizer = optim.Adam(model.parameters(), lr=0.01) 

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()  # 기울기 초기화
    hypothesis = model(x)  # hypothesis 가설
    loss = criterion(hypothesis, y)  # 위에서 정의한 MSELoss
    loss.backward()  # 기울기 계산
    optimizer.step()  # 가중치 수정
    return loss.item()  
'''
.item(): 손실 함숫값과 같이 숫자가 하나인 텐서를 텐서가 아닌 값으로 만들어 준다.
.numpy(): 넘파이 배열로 변환한다. 
'''

epochs = 0
while True:
    epochs += 1
    loss = train(model, criterion, optimizer, x, y)  # 현재는 data loader 쓰지 않는 상태
    print(f'epoch : {epochs}, loss : {loss}')
    time.sleep(0.1)
    if loss < 0.0001: 
        break

#4. evaluate, predict
def evaluate(model, criterion, x, y):
    model.eval()  # model.eval(): model의 모든 레이어가 evaluation mode에 들어간다. 학습할 때만 필요한 dropout, batchnorm 등의 기능을 비활성화한다.
    with torch.no_grad():  # torch.no_grad(): autograd engine(gradient를 계산하는 context)을 비활성화하여 필요한 메모리는 줄여주고 연산속도를 증가시킨다.
        y_pred = model(x)
        loss = criterion(y_pred, y)
    return loss.item()  

loss2 = evaluate(model, criterion, x, y)
print(f'loss : {loss}')  # loss : 9.975879947887734e-05

result = model(torch.Tensor([[4]]))
print('torch.Tensor([[4]]): ', torch.Tensor([[4]]))  #  tensor([[4.]])
print(f'4의 예측값 : {result.item()}')  # 4의 예측값 : 3.979031801223755