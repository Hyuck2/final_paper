import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import torch
import cnn_cpp

class cnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_00 = torch.nn.Conv2d(1, 32, (3, 3), padding=1)
        self.relu_00 = torch.nn.ReLU()
        self.conv2d_01 = torch.nn.Conv2d(32, 32, (3, 3), stride=2, padding=1)
        self.relu_01 = torch.nn.ReLU()
        self.conv2d_02 = torch.nn.Conv2d(32, 64, (3, 3), padding=1)
        self.relu_02 = torch.nn.ReLU()
        self.conv2d_03 = torch.nn.Conv2d(64, 64, (3, 3), stride=2, padding=1)
        self.relu_03 = torch.nn.ReLU()
        self.conv2d_04 = torch.nn.Conv2d(64, 128, (3, 3), padding=1)
        self.relu_04 = torch.nn.ReLU()
        self.conv2d_05 = torch.nn.Conv2d(128, 128, (3, 3), stride=2, padding=1)
        self.relu_05 = torch.nn.ReLU()
        self.conv2d_06 = torch.nn.Conv2d(128, 256, (3, 3), padding=1)
        self.relu_06 = torch.nn.ReLU()
        self.conv2d_07 = torch.nn.Conv2d(256, 256, (3, 3), stride=2, padding=1)
        self.relu_07 = torch.nn.ReLU()
        self.conv2d_08 = torch.nn.Conv2d(256, 512, (3, 3), padding=1)
        self.relu_08 = torch.nn.ReLU()
        self.conv2d_09 = torch.nn.Conv2d(512, 512, (3, 3), stride=2, padding=1)
        self.relu_09 = torch.nn.ReLU()
        self.linear_00 = torch.nn.Linear(512, 50)
        self.relu_10 = torch.nn.ReLU()
        self.linear_01 = torch.nn.Linear(50, 10)
        torch.nn.Softmax(dim=-1)

    def forward(self, x, parameter=[]):
        assert x.dim() > 2
        if x.dim() == 3:
            x = x.view(-1, 1, x.size(-2), x.size(-1))
        return cnn_cpp.forward(x, parameter)