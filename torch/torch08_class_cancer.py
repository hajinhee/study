from turtle import forward
from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from icecream import ic

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

# numpy -> tensor
x_train = torch.FloatTensor(x_train).to(DEVICE)  #  torch.Size([455, 30])
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)  # torch.Size([455, 1])
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
ic(x_train.shape, y_train.shape)  

#2. modeling
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, input_size):  
        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x

model = Model(input_dim=30, output_dim=1).to(DEVICE)

#3. compile, train
criterion = nn.BCELoss()  # binary_crossentropy
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
    print(f'epoch: {epoch}, loss:{loss:.8f}')  # epoch: 1200, loss:0.21978039
    
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
print(f'loss: {loss}')  # loss: 1.049900770187378

y_predict = (model(x_test) >= 0.5).float() 
print('y_predict: ', y_predict[:10])
'''
tensor([[1.],
        [1.],
        [1.],
        [1.],
        [1.],
        [0.],
        [0.],
        [1.],
        [1.],
        [1.]]
'''

acc_score = (y_predict == y_test).float().mean()    
print(f'acc_score : {acc_score:.4f}')  # acc_score : 0.9825

acc_score2 = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy()) 
print(f'acc_score2 : {acc_score2:.4f}')  # acc_score2 : 0.9825
