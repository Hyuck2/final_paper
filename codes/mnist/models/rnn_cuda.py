import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import torch

class rnn(torch.nn.Module):
    def __init__(self):
        super().__init__()