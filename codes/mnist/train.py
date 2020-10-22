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
from models.rnn import rnn
from models.rnn_cpp import rnn as rnn_cpp
from models.rnn_cuda import rnn as rnn_cuda

def argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', default=False)
    p.add_argument('--n_epoch', type=int, default=20)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--verbose', type=bool, default=False)
    p.add_argument('--model', default='fc')
    p.add_argument('--model_dict', type=str)
    p.add_argument('--check', type=bool, default=False)
    p.add_argument('--profile', type=bool, default=False)
    config = p.parse_args()
    return config

def get_model(config):
    if config.model == 'cnn':
        return cnn()
    elif config.model == 'cnn_cpp':
        return cnn_cpp()
    elif config.model == 'cnn_cuda':
        return cnn_cuda()
    elif config.model == 'fc_cpp':
        return fc_cpp()
    elif config.model == 'fc_cuda':
        return fc_cuda()
    elif config.model == 'rnn':
        return rnn()
    elif config.model == 'rnn_cpp':
        return rnn_cpp()
    elif config.model == 'rnn_cuda':
        return rnn_cuda()
    else:
        return fc()

if __name__ == "__main__":
    config = argparser()
    if config.gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    # model save and load function needed
    '''
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
    elif config.model == 'rnn':
        model = rnn().to(device)
    elif config.model == 'rnn_cpp':
        model = rnn_cpp().to(device)
    elif config.model == 'rnn_cuda':
        model = rnn_cuda().to(device)
    else:
        model = fc().to(device)
    '''
    model = get_model(config).to(device)
    if config.check:
        print("Number of Parameters : " + str(len(list(model.parameters()))))
        #print(type(model.parameters()))
        for par in list(model.parameters()):
            print("Size of parameter : " + str(len(par)))
            #print(type(par))
        print("weight")
        print(list(model.parameters())[0][0])
        print("bias")
        print(list(model.parameters())[1][0])

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