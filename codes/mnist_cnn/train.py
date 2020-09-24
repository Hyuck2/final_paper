import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import torch.cuda.profiler as profiler
import pyprof

def argparser(): # arguments
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=bool, default=False)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--n_epochs', type=int)
    p.add_argument('--cpp', type=bool, default=False)
    p.add_argument('--cuda', type=bool, default=False)
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--model_pth', type=str, default='cnn_pytorch.pth')
    config = p.parse_args()
    return config

if __name__ == "__main__":
    config = argparser()