import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
import time
import argparse
from tqdm import tqdm
import torch
import torch.cuda.profiler as profiler
import pyprof

# import models
from models.mnist_cnn import CNN # CNN based on python and pytorch
from models.mnist_cnn_cpp_cu import CNN_cpp, CNN_cu # CNN based on cpp and cu

# import functions
from functions.utils import logger, loader 

def argparser(): # arguments
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=bool, default=False)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--n_epochs', type=int)
    p.add_argument('--learning_rate', type=int, default=0.001)
    p.add_argument('--lang', type=int, default='python')
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--model', type=str, default='cnn')
    p.add_argument('--model_pth', type=str, default='cnn_pytorch.pth')
    config = p.parse_args()
    return config

if __name__ == "__main__":
    config = argparser()
    device = 'cuda' if config.gpu else 'cpu'
    log = logger([config.lang, device, config.model, config.n_epoch])
    data = loader()

    for _ in tqdm(range(config.n_epoch)):
        start = time.time()
        
        end = time.time()
        forward = end - start
        
        start = time.time()
        
        end = time.time()
        backward = end - start
        
        if config.verbose:
            print()
    
    logger.write()
    logger.close()