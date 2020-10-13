import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import torch

class ConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        y = self.layers(x)
        return y
class cnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.Sequential(
            ConvolutionBlock(1, 32),
            ConvolutionBlock(32, 64),
            ConvolutionBlock(64, 128),
            ConvolutionBlock(128, 256),
            ConvolutionBlock(256, 512),
        )
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(512, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x, parameter=[]):
        assert x.dim() > 2
        if x.dim() == 3:
            x = x.view(-1, 1, x.size(-2), x.size(-1))
        z = self.blocks(x)
        y = self.layers(z.squeeze())
        return y