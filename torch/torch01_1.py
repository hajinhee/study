# keras 01과 병렬적으로 진행된다.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim     # optimizer
import torch.nn.functional as F # Loss같은거 기능들? 들어가 있다.
import time

#1. 데이터 정제해서 값 도출
x =  np.array([1,2,3])
y =  np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1)    # torch는 이런식으로 데이터를 받는다.
y = torch.FloatTensor(y).unsqueeze(1)   # unsqueeze 데이터의 차원을 늘려준다. (3,) -> (3,1)

#print(x,y)                 # tensor([1., 2., 3.]) tensor([1., 2., 3.])
#print(x.shape, y.shape)    # torch.Size([3]) torch.Size([3])
# 위의 (3)인 data shape가 (3,1)로 바뀌어야 한다.

#2. 모델구성
# model.add(Dense(1, input_dim=1))      # Dense 밀집
model = nn.Linear(1, 1)         # torch는 input이 앞에간다. 인풋, 아웃풋
# nn.Linear(1,1)에서 앞의 1은 input data의 (3,1)의 1을 의미. 뒤의 1은 y값 (3,1)의 1을 의미

#3. 컴파일, 훈련      
# model.compile(loss='mse', optimizer='adam')
criterion = nn.MSELoss()    # Loss정의
optimizer = optim.Adam(model.parameters(), lr=0.01) # 뒤의 model.parameters() 넣어주면model이 상속받는다
# print(optimizer)


#위의 데이터들로 훈련하겠다 fit. 
# model.fit(x, y, epochs=100, batch_size=1) # batch_size 개념은 들어가지도 않았다 아직. 

def train(model, criterion, optimizer, x, y):
    #model.train()   # 훈련모드
    optimizer.zero_grad()   # 미분값이 중복되지않기 위해, 기울기 초기화
    
    hypothesis = model(x)   # model에 x를 넣겠다. tensor1에서 공부한 추론개념
    
    loss = criterion(hypothesis, y) # 위에서정의한대로 mse가 된다.
    
    loss.backward()     # 기울기 계산. 미분,편미분,합성미분,Gradient descent 공부한 그부분
    optimizer.step()    # 가중치 수정
    return loss.item()  # tensor형태가 아닌 사람이 보기좋은 numpy형태로 값을 반환해준다.

# 위의 함수가 순전파, 역전파를 풀어놓은 개념이다.

# epochs = 200
# for epoch in range(1, epochs+1):
#     loss = train(model,criterion,optimizer,x,y)    # 현재는 data loader 쓰지 않는 상태
#     print(f'epoch : {epoch}, loss : {loss}')
#     time.sleep(0.1)

epochs = 0
while True:
    epochs += 1
    loss = train(model,criterion,optimizer,x,y)    # 현재는 data loader 쓰지 않는 상태
    print(f'epoch : {epochs}, loss : {loss}')
    time.sleep(0.1)
    if loss < 0.0001: break
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

# result = model.predict([4]) # 새로운 x값을 predcit한 결과 
# print('4의 예측값 : ', result)

result = model(torch.Tensor([[4]]))
print(f'4의 예측값 : {result.item()}')