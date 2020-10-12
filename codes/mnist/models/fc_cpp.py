import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import torch
import fc_cpp

class fc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight= torch.nn.Parameter(torch.empty(784,500,26))
        self.bias = torch.nn.Parameter(torch.empty(500,26))
        self.linear_00 = torch.nn.Linear(784, 500) # weight 500, bias 500


    def forward(self, x):
        return fc_cpp.forward(x, self.weight, self.bias)
    