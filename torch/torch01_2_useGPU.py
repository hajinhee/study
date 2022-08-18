# gpu
# criterion 이슈

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim     # optimizer
import torch.nn.functional as F # Loss같은거 기능들? 들어가 있다.
import time

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# torch 버전체크, GPU사용 가능 여부 확인
print(f'torch : {torch.__version__}, 사용DEVICE : {DEVICE}') # torch : 1.10.2+cu113, 사용DEVICE : cuda

# GPU 이름 체크(cuda:0에 연결된 그래픽 카드 기준)
print(torch.cuda.get_device_name()) # NVIDIA GeForce RTX 2080

# 사용 가능 GPU 개수 체크
print(torch.cuda.device_count()) # 1

### ==========================================================================================================

#1. 데이터 정제해서 값 도출
x =  np.array([1,2,3])
y =  np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)   
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)   

#2. 모델구성
model = nn.Linear(1, 1).to(DEVICE)      # GPU 쓰겠다.

#3. 컴파일, 훈련      
criterion = nn.MSELoss()    # criterion은 class가 생성한 객체 instance이다.
optimizer = optim.Adam(model.parameters(), lr=0.01) 

def train(model, criterion, optimizer, x, y):
    #model.train()   # 훈련모드
    optimizer.zero_grad()   
    
    hypothesis = model(x)  
    
    # loss = criterion(hypothesis, y) 
    # loss = nn.MSELoss(hypothesis,y) --> 이렇게하면 바로 에러터진다. 파이썬 문법에대한 이해 필요.
                                        # 생성된 instance안에서만 값을 받을수 있다.
    # loss = nn.MSELoss()(hypothesis,y)   # 이렇게 선언하면 돌아간다.
    loss = F.mse_loss(hypothesis,y)     # 이것도 잘 돌아간다.
    loss.backward()    
    optimizer.step()    
    return loss.item()  

epochs = 0
while True:
    epochs += 1
    loss = train(model,criterion,optimizer,x,y)    # 현재는 data loader 쓰지 않는 상태
    print(f'epoch : {epochs}, loss : {loss}')
    # time.sleep(0.1)
    if loss == 0: break
print('==================================================================================')

#4. 평가, 예측
# loss = model.evaluate(x, y) # 평가하다.

def evaluate(model, criterion, x, y):
    model.eval()        # torch는 eval로 평가한다. 평가모드

    with torch.no_grad():   # gradient를 갱신하지않겠다.
        predict = model(x)
        loss2 = criterion(predict, y)
    return loss2.item()

loss2 = evaluate(model, criterion, x, y)
print(f'최종 loss : {loss2}')

result = model(torch.Tensor([[4]]).to(DEVICE))  # 여기도 to DEVICE해줘야 같은 gpu에서 연산되서 에러안뜬다.
print(f'4의 예측값 : {result.item()}')