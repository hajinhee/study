import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim  
import torch.nn.functional as F 
import time
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# torch 버전체크, GPU사용 가능 여부 확인
print(f'torch : {torch.__version__}, 사용DEVICE : {DEVICE}') # torch : 1.10.2+cu113, 사용DEVICE : cuda
# GPU 이름 체크(cuda:0에 연결된 그래픽 카드 기준)
print(torch.cuda.get_device_name()) # NVIDIA GeForce RTX 2080
# 사용 가능 GPU 개수 체크
print(torch.cuda.device_count()) # 1

#1. data
x =  np.array([1,2,3])  # (3,)
y =  np.array([1,2,3])  # (3,)
x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)  # tensor 변수를 device가 지정한 GPU에 복사한 다음, 연산을 GPU에서 한다는 뜻
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)   

#2. modeling
model = nn.Linear(1, 1).to(DEVICE)  # nn.Linear(input_dim, output_dim)

#3. compile, train      
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01) 

def train(model, optimizer, x, y):
    optimizer.zero_grad()     
    hypothesis = model(x)  # 가설 정의
    # loss = criterion(hypothesis, y) 
    loss = F.mse_loss(hypothesis, y)  
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