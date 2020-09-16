import sys
print(sys.path)
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from model_lltm_extension import LLTM


batch_size = 16
input_features = 32
state_size = 128

X = torch.randn(batch_size, input_features)
h = torch.randn(batch_size, state_size)
C = torch.randn(batch_size, state_size)

rnn = LLTM(input_features, state_size)

forward  = 0
backward = 0

for _ in range(100000):
    start = time.time()
    new_h, new_C = rnn(X, (h, C))
    forward += time.time() - start
    start = time.time()
    (new_h.sum() + new_C.sum()).backward()
    backward += time.time() - start

print('C++ forward and backward')
print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))