from sklearn.datasets import load_iris
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

datasets = load_iris()

x = datasets.data
y = datasets.target

# print(np.unique(y,return_counts = True))  (array([0, 1, 2]), array([50, 50, 50], dtype=int64))

x = torch.FloatTensor(x)
y = torch.LongTensor(y)         # <-- y값 자체가 int값인데 FloatTensor하면 에러가 뜬다. LongTensor하자

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
         train_size = 0.7, shuffle = True, random_state = 66)

x_train = torch.FloatTensor(x_train)#.to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test)#.to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, y_train.shape)   # (105, 4) torch.Size([105, 1])
# print(type(x_train),type(y_train))  # <class 'numpy.ndarray'> <class 'torch.Tensor'>

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

#2. 모델
model = nn.Sequential(
    nn.Linear(4,32),
    nn.ReLU(),
    nn.Linear(32,32),
    nn.ReLU(),
    nn.Linear(32,16),
    nn.ReLU(),
    nn.Linear(16,3),
    # nn.Sigmoid()      torch에서는 마지막에 sofrmax를 안한다.
).to(DEVICE)

#3. 컴파일, 훈련
# criterion = nn.BCELoss()    # Binary CrossEntropy 
criterion = nn.CrossEntropyLoss()   

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
early_stopping = 0
best_loss = 1000
while True:
    epoch += 1
    loss = train(model, criterion, optimizer, x_train, y_train)
    
    print(f'epoch: {epoch}, loss:{loss:.8f}')
        
    if loss < best_loss: 
        best_loss = loss    
        early_stopping = 0
    else:
        early_stopping += 1
    
    if early_stopping == 20: break

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

y_predict = torch.argmax(model(x_test),1)
print(y_predict)

score = (y_predict == y_test).float().mean()    
# y_predict와 y_test가 같다면 True or False로 반환되고 거기에 float해서 0과1로 바꿔준 후 평균을 내면 그게 acc다.
print(f'accyracy : {score:.4f}')

from sklearn.metrics import accuracy_score
score2 = accuracy_score(y_test.cpu().numpy(),y_predict.cpu().numpy()) 
# output을 cpu에 올리고 detach하고? numpy해준다.
print(f'accuracy2 : {score2:.4f}')