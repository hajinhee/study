from sklearn.datasets import load_boston
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from icecream import ic

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

# data split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# numpy -> tensor
x_train = torch.FloatTensor(x_train).to(DEVICE)  # torch.Size([404, 13])
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)  # torch.Size([404, 1])
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)
ic(x_train.shape, y_train.shape) 

#2. modeling
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model,self).__init__()  # or  super().__init__()
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
        
model = Model(13, 1).to(DEVICE)

#3. compile, train
criterion = nn.MSELoss()  # mse
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
    print(f'epoch: {epoch}, loss:{loss:.8f}')  # epoch: 1200, loss:5.09792328
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
print(f'loss: {loss}')  # loss: 7.163231372833252

y_predict = model(x_test)
print('y_predict: ', y_predict[:10])
'''
tensor([[12.1407],
        [44.5078],
        [24.8345],
        [55.5364],
        [19.0945],
        [22.6870],
        [20.2494],
        [24.7807],
        [46.6086],
        [14.8737]]
'''
score = mean_squared_error(y_test.cpu().numpy(), y_predict.cpu().detach().numpy())
print(f'score : {score:.4f}')  # score : 7.1632
r2 = r2_score(y_test.cpu().numpy(),y_predict.cpu().detach().numpy())
print(f'r2_scrore:{r2:.4f}')  # r2_scrore:0.9143