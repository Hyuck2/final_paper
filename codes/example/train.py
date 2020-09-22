import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from model_lltm import LLTM
from model_lltm_extension import LLTM as LLTM_cpp
from model_lltm_cuda import LLTM as LLTM_cuda

def argparser(): # arguments
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', default=False)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--input_features', type=int, default=32)
    p.add_argument('--state_size', type=int, default=128)
    p.add_argument('--n_epochs', type=int, default=100000)
    p.add_argument('--cpp', default=False)
    p.add_argument('--cuda', default=False)
    p.add_argument('--verbose', default=True)
    p.add_argument('--model', default='lltm')
    config = p.parse_args()
    return config

if __name__ == "__main__":
    config = argparser()
    batch_size = config.batch_size
    input_features = config.input_features
    state_size = config.state_size
    string = ''

    if config.gpu:
        string = 'CUDA'
        d = torch.device('cuda:0')
    else:
        string = 'CPU'
        d = torch.device('cpu')
    
    X = torch.randn(batch_size, input_features, device=d)
    h = torch.randn(batch_size, state_size, device=d)
    C = torch.randn(batch_size, state_size, device=d)

    if config.cpp:
        string = 'Cpp_' + string
        rnn = LLTM_cpp(input_features, state_size).to(d)
    elif config.cuda:
        string = 'Cpp_Custom_CUDA_' + string
        rnn = LLTM_cuda(input_features, state_size).to(d)
    else:
        string = 'Python_' + string
        rnn = LLTM(input_features, state_size).to(d)

    forward  = 0
    backward = 0
    string = './' + string +'_' +str(config.n_epochs) + '.log'
    print(string)


    log_file = open(string, 'a')
    for _ in tqdm(range(config.n_epochs)):
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        forward += time.time() - start
        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        backward += time.time() - start
    if not config.verbose:
        print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e5, backward * 1e6/1e5))
    log_file.write('Forward: {:.3f} us | Backward {:.3f} us\n'.format(forward * 1e6/1e5, backward * 1e6/1e5))
    log_file.close()