from sklearn.datasets import load_iris
import torch
import torch.nn as nn
import torch.optim as optim
from icecream import ic
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. data
datasets = load_iris()
x = datasets.data
y = datasets.target
ic(np.unique(y))  # array([0, 1, 2]) -> 다중분류

# data split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# numpy -> tensor
x_train = torch.FloatTensor(x_train).to(DEVICE)  # torch.Size([120, 4])
y_train = torch.LongTensor(y_train).to(DEVICE)  # torch.Size([120])
x_test = torch.FloatTensor(x_test).to(DEVICE)  # torch.Size([30, 4])
y_test = torch.LongTensor(y_test).to(DEVICE)  # torch.Size([30])

#2. modeling
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        #super().__init__()
        super(Model,self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        out = self.linear4(x)
        return out    
        
model = Model(4, 3).to(DEVICE)

#3. compile, train
criterion = nn.CrossEntropyLoss()  # categorical_crossentropy  
optimizer = optim.Adam(model.parameters(), lr=0.1) 
def train(model, criterion, optimizer, x_train, y_train):
    model.train()  
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()                
    return loss.item()

epoch = 0
while True:
    epoch += 1
    early_stopping = 0
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f'epoch: {epoch}, loss:{loss:.8f}')  # epoch: 1200, loss:0.00000283
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
print(f'loss: {loss}')  # loss: 2.552999973297119

y_predict = torch.argmax(model(x_test), 1)
print('y_predict: ', y_predict)
'''
tensor([1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 2, 2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 1, 2, 1, 2, 0, 1, 1, 2], device='cuda:0')
'''
acc_score = (y_predict == y_test).float().mean()    
print(f'acc_score : {acc_score:.4f}')  # acc_score : 0.9000

acc_score2 = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())  # accuracy_score() 사용
print(f'acc_score2 : {acc_score2:.4f}')  # acc_score2 : 0.9000

'''
****RuntimeError: 0D or 1D target tensor expected, multi-target not supported 에러****
nn.CrossEntropyLoss() 혹은 F.cross_entropy 를 사용했을 때 나타나는 에러이다.
nn.CrossEntropyLoss()(pred, target) 이렇게 계산이 되는데
가령 pred의 shape의 [B, C]라면 target의 shape은 [B]가 되어야 하는데 [B, 1] 이렇게 돼서 문제가 발생한다.
문제를 해결하려면 target의 shape를 축소한다. [B, 1] -> [B]
'''