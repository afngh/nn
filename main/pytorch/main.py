import torch
import torch.nn as nn
import torch.optim as optim

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2,2)
        self.l2 = nn.Linear(2,1)
        
    def forward(self,x):
        x = torch.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        return x

model = NN()

X = torch.tensor([
        [0.,0.],
        [0.,1.],
        [1.,0.],
        [1.,1.],
    ])

y = torch.tensor([
    [0],
    [1],
    [1],
    [0],
])

output = model.forward(X)

print(output)