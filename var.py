import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
# Define the neural network
class GIKApproxNet(nn.Module):
    def __init__(self):
        super(GIKApproxNet, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
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


def rand(N, M):
    return (torch.rand(N, 1) - 0.5) * 2 * M

def func(mu_square, a_ik):
    """
    mu : (B, 2)
    a_ik : (B, 1)
    """
    A = mu_square
    B = torch.sqrt(a_ik)
    return torch.exp(-A * B * 0.5)

# Initialize the model, loss function, and optimizer
model = GIKApproxNet().cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

B = 5000000
iteration = 20000
pbar = trange(iteration)
for _ in pbar:
    mu_square = (rand(2*B, 100).reshape(-1, 2).cuda()).square().sum(dim=-1, keepdim=True)
    a_ik = 1 / (rand(B, 10) ** 2 + 1).square().cuda()
    input_ = torch.cat([mu_square, a_ik], dim=-1) # (B, 2)
    pred = model(input_) # (B, 1)
    optimizer.zero_grad()
    loss = criterion((pred * a_ik).sum(-1, keepdim=True), func(mu_square, a_ik))
    pbar.set_postfix({"loss":loss.item()})
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "var.pt")

# model.load_state_dict(torch.load("var.pt"))
eval_B = 10

with torch.no_grad():
    mu_square = (rand(2*eval_B, 100).reshape(-1, 2).cuda()).square().sum(dim=-1, keepdim=True)
    a_ik = 1 / (rand(eval_B, 10) ** 2 + 1).square().cuda()
    input_ = torch.cat([mu_square, a_ik], dim=-1) # (B, 5)
    pred = model(input_) # (B, 2)
    print(torch.cat([(pred * a_ik).sum(-1, keepdim=True), func(mu_square, a_ik), (pred * a_ik).sum(-1, keepdim=True)-func(mu_square, a_ik)], dim=-1))
