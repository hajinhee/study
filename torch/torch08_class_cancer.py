from turtle import forward
from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

datasets = load_breast_cancer()

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
# model = nn.Sequential(
#     nn.Linear(30,32),
#     nn.ReLU(),
#     nn.Linear(32,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,1),       
#     nn.Sigmoid()
# ).to(DEVICE)

class Yeram_Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        # super().__init__(),                      # 이 아래의 형식과 같은 의미
        super(Yeram_Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, input_size):     # 이걸 사용하기 위해 nn.Module을 상속받았다
        x = self.linear1(input_size)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x

model = Yeram_Model(30,1).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss()    # Binary CrossEntropy

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

y_predict = (model(x_test) >= 0.5).float()  # argmax가 이전에 0과1로 바꿔줬던것처럼 여기선 이런 방식으로 바꿔본다.
print(y_predict[:10])

score = (y_predict == y_test).float().mean()    
# y_predict와 y_test가 같다면 True or False로 반환되고 거기에 float해서 0과1로 바꿔준 후 평균을 내면 그게 acc다.
print(f'accyracy : {score:.4f}')

from sklearn.metrics import accuracy_score
score2 = accuracy_score(y_test.cpu().numpy(),y_predict.cpu().numpy()) 
# output을 cpu에 올리고 detach하고? numpy해준다.
print(f'accuracy2 : {score2:.4f}')

# ================== 평가 예측 ====================
# loss: 0.7141337990760803
# tensor([[1.],
#         [1.],
#         [1.],
#         [1.],
#         [1.],
#         [0.],
#         [0.],
#         [1.],
#         [1.],
#         [1.]], device='cuda:0')
# accyracy : 0.9825
# accuracy2 : 0.9825