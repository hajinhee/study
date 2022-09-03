from sklearn.datasets import load_boston
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from icecream import ic
from sklearn.metrics import mean_squared_error,r2_score
warnings.filterwarnings(action='ignore')

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

# TensorDataset()으로 데이터 결합 
train_set = TensorDataset(x_train, y_train)  # train_set --> x_train, y_train의 각 0행이 묶여있다.
test_set = TensorDataset(x_test, y_test)  # test_set

# DataLoader()로 합쳐진 데이터를 tensor에서 사용가능한 형태로 불러온다.                    
train_loader = DataLoader(train_set, batch_size=36, shuffle=True)
test_loader = DataLoader(test_set, batch_size=36, shuffle=False)

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
model = Model(13, 1).to(DEVICE)

#3. compile, train
criterion = nn.MSELoss()  # mse
optimizer = optim.Adam(model.parameters(), lr=0.01)  
def train(model, criterion, optimizer, train_loader):
    model.train()   
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch) 
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

epoch = 0
early_stopping = 0
best_loss = 10000
while True:
    epoch += 1
    loss = train(model, criterion, optimizer, train_loader)
    print(f'epoch: {epoch}, loss:{loss:.8f}')  # epoch: 1008, loss:0.59948073
    if loss < best_loss: 
        best_loss = loss    
        early_stopping = 0
    else:
        early_stopping += 1

    if early_stopping == 100:
         break

#4. evaluate, predict
def evaluate(model, criterion, loader):
    model.eval()    
    total_loss = 0
    for x_batch,y_batch in loader:
        with torch.no_grad():
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)
loss = evaluate(model, criterion, test_loader)
print(f'loss: {loss}')  # loss: 9.948384284973145

y_predict = model(x_test)    
mse_loss = mean_squared_error(y_test.cpu().numpy(), y_predict.cpu().detach().numpy())  # mean_squared_error()
print(f'mse_loss : {mse_loss:.4f}')  # mse_loss : 9.8422

r2 = r2_score(y_test.cpu().numpy(), y_predict.cpu().detach().numpy())  # r2_score()
print(f'r2_scrore: {r2:.4f}')  # r2_scrore: 0.8822