import torch
import torch.nn as nn
import torch.cuda.profiler as profiler
import pyprof

class RecurrentClassifier(nn.module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__()
        