from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from icecream import ic
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. data
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# data split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# numpy.ndarray -> tensor 
x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)  # torch.Size([455, 1])
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)  # torch.Size([114, 1])

#2. modeling
model = nn.Sequential(
    nn.Linear(30,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,1),     
    nn.Sigmoid()
).to(DEVICE)

#3. compile, train
criterion = nn.BCELoss()  # binary_crossentropy 
optimizer = optim.Adam(model.parameters(), lr=0.1)  # model.parameters() 갱신하는 것이 곧 weight와 bias 갱신하는 것과 같은 의미
def train(model, criterion, optimizer, x_train, y_train):
    model.train() 
    optimizer.zero_grad()  # 기울기 초기화
    hypothesis = model(x_train)  # 가설 정의
    loss = criterion(hypothesis, y_train)  # 여기까지 순전파
    loss.backward()  # 역전파
    optimizer.step()                
    return loss.item()

epoch = 0
while True:
    epoch += 1
    early_stopping = 0
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f'epoch: {epoch}, loss:{loss:.8f}')  # epoch: 1200, loss:0.65934068
    if epoch == 1200:
        break

#4. evaluate, predict
def evaluate(model, criterion, x_test, y_test):
    model.eval() 
    with torch.no_grad():  
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        return loss.item()

loss = evaluate(model, criterion, x_test, y_test)
print(f'loss: {loss}')  # loss: 1.9740873575210571

y_predict = (model(x_test) >= 0.5).float()
'''
tensor([[1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [0.],
        [0.],
        [1.], ......  
'''
acc_score = (y_predict == y_test).float().mean()    
print(f'acc_score : {acc_score:.4f}')  # acc_score : 0.9649
acc_score2 = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy()) 
print(f'acc_score2 : {acc_score2:.4f}')  # acc_score2 : 0.9649