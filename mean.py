import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
# Define the neural network
class GIKApproxNet(nn.Module):
    def __init__(self):
        super(GIKApproxNet, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5) 
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  
        x = torch.relu(self.fc2(x))
        x = self.dropout(x) 
        x = torch.relu(self.fc3(x))
        x = self.dropout(x) 
        x = torch.relu(self.fc4(x))
        x = self.dropout(x) 
        x = self.fc5(x)
        return x


# Initialize the model, loss function, and optimizer
model = GIKApproxNet().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
def func(mu, d, x, var):
    """
    mu : (B, 2)
    d : (B, 2)
    x : (B, 2)
    var: (B, 1)
    """
    A = (d * mu).sum(-1, keepdim=True) ** 2
    B = (d * x).sum(-1, keepdim=True) * (d * mu).sum(-1, keepdim=True)
    C = (x * x).sum(-1, keepdim=True)
    return torch.exp(-(A - 2 * B + C) / (2 * var))

def rand(N, M):
    return (torch.rand(N, 1) - 0.5) * 2 * M

B = 5000000
iteration = 20000
for _ in trange(iteration):
    theta = rand(B, torch.pi).cuda()
    d = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1).cuda()
    x = rand(2*B, 100).reshape(-1, 2).cuda()
    var = (rand(B, 10) ** 2 + 1).cuda()
    mu = rand(2*B, 100).reshape(-1, 2).cuda()
    input_ = torch.cat([d, x, var], dim=-1) # (B, 5)
    pred = model(input_) # (B, 2)
    optimizer.zero_grad()
    loss = criterion((pred * mu).sum(-1, keepdim=True), func(mu, d, x, var))
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "mean.pt")

eval_B = 10
with torch.no_grad():
    theta = rand(10, torch.pi).cuda()
    d = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1).cuda()
    x = rand(2*eval_B, 100).reshape(-1, 2).cuda()
    var = (rand(eval_B, 10) ** 2 + 1).cuda()
    mu = rand(2*eval_B, 100).reshape(-1, 2).cuda()
    input_ = torch.cat([d, x, var], dim=-1) # (B, 5)
    pred = model(input_) # (B, 2)
    print(torch.cat([(pred * mu).sum(-1, keepdim=True), func(mu, d, x, var), (pred * mu).sum(-1, keepdim=True)-func(mu, d, x, var)], dim=-1))
