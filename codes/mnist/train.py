import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from models.cnn_cpp import cnn as cnn_cpp
from models.cnn_cuda import cnn as cnn_cuda
from models.cnn import cnn
from models.fc import fc
from models.fc_cpp import fc as fc_cpp
from utils.utils import load_mnist
from functions.trainer import trainer

def argparser(): # arguments
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', default=False)
    p.add_argument('--n_epoch', type=int, default=10)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--verbose', default=True)
    p.add_argument('--model', default='fc')
    config = p.parse_args()
    return config

if __name__ == "__main__":
    # configurations
    config = argparser()
    
    if config.gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    if config.model == 'cnn':
        model = cnn().to(device)
    elif config.model == 'cnn_cpp':
        model = cnn_cpp().to(device)
    elif config.model == 'cnn_cuda':
        model = cnn_cuda().to(device)
    elif config.model == 'fc_cpp':
        model = fc_cpp().to(device)
    else:
        model = fc().to(device)
    print(len(list(model.parameters())))
    for par in list(model.parameters()):
        print(len(par))

    # data loader
    x, y = load_mnist(is_train=True)
    x = x.view(x.size(0), -1)
    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt
    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0)

    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()
    trainer = trainer(model, config, optimizer, crit)
    trainer.train((x[0], y[0]), (x[1], y[1]))

    if not config.verbose:
        pass