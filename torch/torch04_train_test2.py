import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

#1. data
x = np.array(range(100))  # (100, )
y = np.array(range(1,101))  # (100, )

# data split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

x_train = torch.FloatTensor(x_train).unsqueeze(1).to(DEVICE)  # torch.Size([80, 1])
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)  # torch.Size([20, 1])
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)  # torch.Size([20, 1])
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)  # torch.Size([80, 1])

#2. modeling
model = nn.Sequential(
    nn.Linear(1,5),
    nn.Linear(5,4),
    nn.Linear(4,3),
    nn.ReLU(),
    nn.Linear(3,4),
    nn.Linear(4,1),     
).to(DEVICE)

#3. compile, train
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1)
def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    hyporthesis = model(x)
    loss = criterion(hyporthesis, y)
    loss.backward()  
    optimizer.step() 
    return loss.item()

EPOCHS = 100
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch : ', epoch, 'loss : ', loss)  # epoch :  100 loss :  0.09630274027585983
    
# evaluate, predict
def evaluate(model, criterion, x, y):
    model.eval()
    with torch.no_grad():
        predict = model(x)
        loss2 = criterion(predict, y)
        return loss2.item()    

loss = evaluate(model, criterion, x_test, y_test)
print('loss: ', loss)  # loss:  0.0804831013083458

result = model(x_test.to(DEVICE))
print(result.cpu().detach().numpy())
'''
[[ 9.278787  ]
 [93.537056  ]
 [ 5.3136926 ]
 [ 6.304966  ]
 [52.894825  ]
 [41.990814  ]
 [ 0.28592545]
 [73.71157   ]
 [88.58068   ]
 [68.75521   ]
 [26.130444  ]
 [19.191523  ]
 [27.121714  ]
 [30.095535  ]
 [66.77266   ]
 [50.912273  ]
 [80.65049   ]
 [45.95591   ]
 [39.017     ]
 [58.842464  ]]
'''
result = model(torch.Tensor(([[11]])).to(DEVICE)).to(DEVICE)
print(result.cpu().detach().numpy())  # [[12.252608]]

