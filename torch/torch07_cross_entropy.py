from sklearn.datasets import load_iris
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from icecream import ic
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. data
datasets = load_iris()
x = datasets.data
y = datasets.target
ic(np.unique(y, return_counts=True))  # ([0, 1, 2]) -> 다중분류

#2. data split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# numpy -> tensor
x_train = torch.FloatTensor(x_train).to(DEVICE)  # torch.Size([120, 4])
y_train = torch.LongTensor(y_train).to(DEVICE)  # torch.Size([120]) -->  LongTensor
x_test = torch.FloatTensor(x_test).to(DEVICE)  # torch.Size([30, 4])
y_test = torch.LongTensor(y_test).to(DEVICE) # torch.Size([30]) -->  LongTensor

#2. modeling
model = nn.Sequential(
    nn.Linear(4,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,3),
).to(DEVICE)

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
early_stopping = 0
best_loss = 1000
while True:
    epoch += 1
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(f'epoch: {epoch}, loss:{loss:.8f}')
    if loss < best_loss:  # best_loss 값을 계속 갱신
        best_loss = loss    
        early_stopping = 0
    else:
        early_stopping += 1
    if early_stopping == 20: 
        break

#4. evaluate, predict
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        return loss.item()
loss = evaluate(model, criterion, x_test, y_test)
print(f'loss: {loss}')  # loss: 0.4629771113395691

y_predict = torch.argmax(model(x_test), 1)  # torch.argmax(input(Tensor), dim(int)): dim은 축소할 차원
print('y_predict: ', y_predict)
'''
tensor([1, 1, 1, 0, 1, 1, 0, 0, 0, 2, 2, 2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 1, 2, 1, 2, 0, 0, 1, 2]
'''

acc_score = (y_predict == y_test).float().mean()    
print(f'accyracy : {acc_score:.4f}')  # accyracy : 0.9667

acc_score2 = accuracy_score(y_test.cpu().numpy(),y_predict.cpu().numpy()) 
print(f'accuracy2 : {acc_score2:.4f}')  # accuracy2 : 0.9667