import sys
sys.path.append('/home/hyuck2/.local/lib/python3.8/site-packages')
sys.path.append('/home/hyuck2/repos/github/final_paper/codes/mnist_cnn/models')
sys.path.append('/home/hyuck2/repos/github/final_paper/codes/mnist_cnn/functions')
import time
import argparse
from tqdm import tqdm
import torch
import torch.cuda.profiler as profiler
import pyprof

# import models
from mnist_cnn import CNN # CNN based on python and pytorch
from mnist_cnn_cpp_cuda import CNN_cpp, CNN_cu # CNN based on cpp and cu

# import functions
from functions.utils import logger, loader
from functions.trainer import Trainer

def argparser(): # arguments
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=bool, default=False)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--learning_rate', type=int, default=0.001)
    p.add_argument('--lang', type=str, default='python')
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--model', type=str, default='cnn')
    p.add_argument('--model_pth', type=str, default='cnn_pytorch.pth')
    p.add_argument('--profile', type=bool, default=False)
    p.add_argument('--train_ratio', type=float, default=.8)
    config = p.parse_args()
    return config

def get_model(config):
    if config.model == 'cnn_cpp':
        pass
    elif config.model == 'cnn_cuda':
        pass
    elif config.model == 'cnn_cpp_cuda':
        pass
    elif config.model == 'cnn':
        model = CNN(10)
    else:
        raise NotImplementedError('You need to specify model name.')

    return model

def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu is False else torch.device('cuda')

    train_loader, valid_loader, test_loader = loader(config)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    model = get_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    crit = torch.nn.CrossEntropyLoss()

    if config.verbose >= 2:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)

if __name__ == "__main__":
    config = argparser()
    main(config)