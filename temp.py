import torch
import math
from torch import matmul as mm
S = torch.tensor([[2, 0],[0, 3.]], requires_grad=True) #(2,2)
R = torch.tensor([[math.cos(math.pi/4), -math.sin(math.pi/4)],[math.sin(math.pi/4), math.cos(math.pi/4)]] , requires_grad=True) #(2,2)
d = torch.tensor([[0.6], [0.8]], requires_grad=True) # (2,1)
RS = mm(R, S)
dRS = mm(d.permute(1, 0), RS)  # (1, 2)
value = mm(dRS, dRS.permute(1, 0))
value.backward()
dR = mm(d.permute(1, 0), R) # (1, 2)
grad = 2 * mm(mm(dR.permute(1, 0), dR), S)
breakpoint()