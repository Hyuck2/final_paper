import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import argparse
import torch
#utilities
from utils.utils import load_mnist
from functions.trainer import trainer
# models
from models.cnn_cpp import cnn as cnn_cpp
from models.cnn_cuda import cnn as cnn_cuda
from models.cnn import cnn
from models.fc import fc
from models.fc_cpp import fc as fc_cpp
from models.fc_cuda import fc as fc_cuda

def argparser(): # arguments
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', default=False)
    p.add_argument('--n_epoch', type=int, default=10)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--verbose', default=False)
    p.add_argument('--model', default='fc')
    p.add_argument('--check', default=False)
    p.add_argument('--profile', default=False)
    config = p.parse_args()
    return config

if __name__ == "__main__":
    config = argparser()
    if config.gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    # model save and load function needed
    if config.model == 'cnn':
        model = cnn().to(device)
    elif config.model == 'cnn_cpp':
        model = cnn_cpp().to(device)
    elif config.model == 'cnn_cuda':
        model = cnn_cuda().to(device)
    elif config.model == 'fc_cpp':
        model = fc_cpp().to(device)
    elif config.model == 'fc_cuda':
        model = fc_cuda().to(device)
    else:
        model = fc().to(device)
    
    if config.check:
        print("Parameters")
        print(len(list(model.parameters())))
        print(type(model.parameters()))
        for par in list(model.parameters()):
            print(len(par))
            print(type(par))
        print("Buffers")
        print(len(list(model.buffers())))
        print(type(model.buffers()))
        for buf in list(model.buffers()):
            print(len(buf))
            print(type(buf))

    else:
        x, y = load_mnist(is_train=True, flatten=False)
        if config.model == 'fc' or config.model == 'fc_cpp' or config.model == 'fc_cuda':
            flatten = True
            x = x.view(x.size(0), -1)
        else:
            flatten = False

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
        
        optimizer = torch.optim.Adam(model.parameters())
        crit = torch.nn.CrossEntropyLoss()
        trainer = trainer(model, config, optimizer, crit)
        if config.profile:
            with torch.autograd.profiler.profile(record_shapes=True) as result:
                with torch.autograd.profiler.record_function("Trainer"):
                    trainer.train((x[0], y[0]), (x[1], y[1]))
            print(result.key_averages().table(sort_by="cpu_time_total", row_limit=10))        
        else:
            trainer.train((x[0], y[0]), (x[1], y[1]))