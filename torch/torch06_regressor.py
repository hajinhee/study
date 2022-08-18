from sklearn.datasets import load_boston
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

datasets = load_boston()

x = datasets.data
y = datasets.target

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
         train_size = 0.7, shuffle = True, random_state = 66)

x_train = torch.FloatTensor(x_train)#.to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)#.to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, y_train.shape)   # (398, 30) torch.Size([398, 1])
# print(type(x_train),type(y_train))  # <class 'numpy.ndarray'> <class 'torch.Tensor'>

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

#2. 모델
model = nn.Sequential(
    nn.Linear(13,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,1),        
    # nn.Sigmoid()
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.MSELoss()     

optimizer = optim.Adam(model.parameters(), lr=0.1)  # 옵티마이저 정의할때 tensorflow에서는 안하지만 
# torch에서는 다르다 model의 parameters를 갱신하는것이 Gradient Descent에서 weight와 bias를 
# 갱신하는것과 같은 의미. 그래서 optim.Adam(model.parameters())해준다.

def train(model, criterion, optimizer, x_train, y_train):
    model.train()   # model.eval()은 역전파 갱신이 안됨. model.train()도 있지만
                    # 항상 default처럼 있기때문에 쓰지않아도 괜찮다.
    
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)   # 여기까지가 순전파
    
    loss.backward()                         # 역전파
    optimizer.step()                
    return loss.item()

epoch = 0
while True:
    epoch += 1
    early_stopping = 0
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f'epoch: {epoch}, loss:{loss:.8f}')
    
    if epoch == 1200:break
    # if :
    #     early_stopping = 0
    
    # else:
    #     early_stopping += 1
    
    # if early_stopping == 20:
    #     break

#4.평가, 예측
print("================== 평가 예측 ====================")
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        return loss.item()

loss = evaluate(model, criterion, x_test, y_test)
print(f'loss: {loss}')


from sklearn.metrics import mean_squared_error,r2_score
y_predict = model(x_test)
# score = mean_squared_error(y_test.cpu().numpy(),y_predict.cpu().detach().numpy())
# print(f'score : {score:.4f}')
r2 = r2_score(y_test.cpu().numpy(),y_predict.cpu().detach().numpy())
print(f'r2_scrore:{r2:.4f}')