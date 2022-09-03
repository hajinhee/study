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
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

# scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# numpy.ndarray -> tensor 
x_train = torch.FloatTensor(x_train).to(DEVICE)  # torch.Size([354, 13])
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)  # torch.Size([354, 1])
x_test = torch.FloatTensor(x_test).to(DEVICE)  # torch.Size([152, 13])
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)  # torch.Size([152, 1])
ic(x_train.size(), x_test.size(), y_train.size())

#2. modeling
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

#3. compile, train
criterion = nn.MSELoss()     
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
    print(f'epoch: {epoch}, loss:{loss:.8f}')  # epoch: 1200, loss:1.41488850
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
print(f'loss: {loss}')  # loss: 10.471636772155762

y_predict = model(x_test)
'''
tensor([[10.5657],
        [49.6757],
        [22.5117],
        [56.1219],
        [20.5614],
        [25.2349],
        [19.8659],
        [17.4919],
        [48.0641],
        [16.8876],
        [23.1622],
        [ 5.3543],
        [32.3307],
        [30.6452],
        [13.9996],
        [20.8643],
        [13.1116],
        [15.8158],
'''
r2 = r2_score(y_test.cpu().numpy(), y_predict.cpu().detach().numpy())
print(f'r2_scrore: {r2:.4f}')  # r2_scrore: 0.8733