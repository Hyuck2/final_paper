import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import torch
import fc_cuda

class fc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_00 = torch.nn.Linear(784, 500)
        self.relu_00 = torch.nn.ReLU()
        self.linear_01 = torch.nn.Linear(500, 400)
        self.relu_01 = torch.nn.ReLU()
        self.linear_02 = torch.nn.Linear(400, 300)
        self.relu_02 = torch.nn.ReLU()
        self.linear_03 = torch.nn.Linear(300, 200)
        self.relu_03 = torch.nn.ReLU()
        self.linear_04 = torch.nn.Linear(200, 100)
        self.relu_04 = torch.nn.ReLU()
        self.linear_05 = torch.nn.Linear(100, 50)
        self.relu_05 = torch.nn.ReLU()
        self.linear_06 = torch.nn.Linear(50, 10)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, parameter = []):
        return fc_cuda.forward(x, parameter)