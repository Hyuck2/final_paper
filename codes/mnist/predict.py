import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import argparse
import torch
import numpy
import matplotlib.pyplot as plt

# models
from models.cnn_cpp import cnn as cnn_cpp
from models.cnn_cuda import cnn as cnn_cuda
from models.cnn import cnn
from models.fc import fc
from models.fc_cpp import fc as fc_cpp
from models.fc_cuda import fc as fc_cuda
from models.rnn import rnn
from models.rnn_cpp import rnn as rnn_cpp
from models.rnn_cuda import rnn as rnn_cuda

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_dict', type=str)
    config = p.parse_args()
    return config

if __name__ == "__main__":
    config = argparser()
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    state_dict = torch.load(config.model_dict, device)
    