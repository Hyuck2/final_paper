import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import torch


class fc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight= torch.nn.Parameter(torch.empty(500,26))
        self.bias = torch.nn.Parameter(torch.empty(500,26))
        self.linear_00 = torch.nn.Linear(784, 500) # weight 500, bias 500
        self.relu_00 = torch.nn.LeakyReLU()
        self.batch_norm_01 = torch.nn.BatchNorm1d(500) 
        self.linear_01 = torch.nn.Linear(500, 400) # weight 400 bias 400
        self.relu_01 = torch.nn.LeakyReLU()
        self.batch_norm_02 = torch.nn.BatchNorm1d(400)
        self.linear_02 = torch.nn.Linear(400, 300)
        self.relu_02 = torch.nn.LeakyReLU()
        self.batch_norm_03 = torch.nn.BatchNorm1d(300)
        self.linear_03 = torch.nn.Linear(300, 200)
        self.relu_03 = torch.nn.LeakyReLU()
        self.batch_norm_04 = torch.nn.BatchNorm1d(200)
        self.linear_04 = torch.nn.Linear(200, 100)
        self.relu_04 = torch.nn.LeakyReLU()
        self.batch_norm_05 = torch.nn.BatchNorm1d(100)
        self.linear_05 = torch.nn.Linear(100, 50) # weight 50, bias 50
        self.relu_05 = torch.nn.LeakyReLU()
        self.batch_norm_06 = torch.nn.BatchNorm1d(50)
        self.linear_06 = torch.nn.Linear(50, 10) # weight 10, bias 10
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        y = self.linear_00(x)
        y = self.relu_00(y)
        y = self.batch_norm_01(y)
        y = self.linear_01(y)
        y = self.relu_01(y)
        y = self.batch_norm_02(y)
        y = self.linear_02(y)
        y = self.relu_02(y)
        y = self.batch_norm_03(y)
        y = self.linear_03(y)
        y = self.relu_03(y)
        y = self.batch_norm_04(y)
        y = self.linear_04(y)
        y = self.relu_04(y)
        y = self.batch_norm_05(y)
        y = self.linear_05(y)
        y = self.relu_05(y)
        y = self.batch_norm_06(y)
        y = self.linear_06(y)
        y = self.softmax(y)
        return y

'''
class fc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_00 = torch.nn.Linear(784, 500) # weight 500, bias 500
        self.relu_00 = torch.nn.LeakyReLU()
        self.batch_norm_01 = torch.nn.BatchNorm1d(500) 
        self.linear_01 = torch.nn.Linear(500, 400) # weight 400 bias 400
        self.relu_01 = torch.nn.LeakyReLU()
        self.batch_norm_02 = torch.nn.BatchNorm1d(400)
        self.linear_02 = torch.nn.Linear(400, 300)
        self.relu_02 = torch.nn.LeakyReLU()
        self.batch_norm_03 = torch.nn.BatchNorm1d(300)
        self.linear_03 = torch.nn.Linear(300, 200)
        self.relu_03 = torch.nn.LeakyReLU()
        self.batch_norm_04 = torch.nn.BatchNorm1d(200)
        self.linear_04 = torch.nn.Linear(200, 100)
        self.relu_04 = torch.nn.LeakyReLU()
        self.batch_norm_05 = torch.nn.BatchNorm1d(100)
        self.linear_05 = torch.nn.Linear(100, 50) # weight 50, bias 50
        self.relu_05 = torch.nn.LeakyReLU()
        self.batch_norm_06 = torch.nn.BatchNorm1d(50)
        self.linear_06 = torch.nn.Linear(50, 10) # weight 10, bias 10
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        y = self.linear_00(x)
        y = self.relu_00(y)
        y = self.batch_norm_01(y)
        y = self.linear_01(y)
        y = self.relu_01(y)
        y = self.batch_norm_02(y)
        y = self.linear_02(y)
        y = self.relu_02(y)
        y = self.batch_norm_03(y)
        y = self.linear_03(y)
        y = self.relu_03(y)
        y = self.batch_norm_04(y)
        y = self.linear_04(y)
        y = self.relu_04(y)
        y = self.batch_norm_05(y)
        y = self.linear_05(y)
        y = self.relu_05(y)
        y = self.batch_norm_06(y)
        y = self.linear_06(y)
        y = self.softmax(y)
        return y
        '''