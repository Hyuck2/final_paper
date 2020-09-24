import torch
import torch.nn
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.cuda.profiler as profiler
import pyprof

def argparser(): # arguments
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=bool, default=False)
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--model_pth', type=str, default='cnn_pytorch.pth')
    config = p.parse_args()
    return config

if __name__ == "__main__":
    config = argparser()