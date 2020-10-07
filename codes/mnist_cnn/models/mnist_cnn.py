import torch

class CNN_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        y = self.layers(x)
        return y


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.Sequential( # |x| = (n, 1, 28, 28)
            CNN_Block(1, 32), # (n, 32, 14, 14)
            CNN_Block(32, 64), # (n, 64, 7, 7)
            CNN_Block(64, 128), # (n, 128, 4, 4)
            CNN_Block(128, 256), # (n, 256, 2, 2)
            CNN_Block(256, 512), # (n, 512, 1, 1)
        )
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(512, 50),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(50),
            torch.nn.Linear(50, 10),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x):
        assert x.dim() > 2
        if x.dim() == 3:
            x = x.view(-1, 1, x.size(-2), x.size(-1))
        z = self.blocks(x)
        y = self.layers(z.squeeze())
        return y