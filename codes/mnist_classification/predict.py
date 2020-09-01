import torch
import torch.nn
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from functions.data_loader import load_mnist
from models.model_fully_connected import FullyConnectedClassifier
from models.model_cnn import ConvolutionalClassifier
from models.model_rnn import RecurrentClassifier
from train import get_model

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_fn', required=True) # Trained model
    p.add_argument('--plot', default=True) # Plot test Image
    config = p.parse_args()
    return config

def load(fn, device):
    d = torch.load(fn, map_location=device)
    return d['config'], d['model']

def _plot(x, y_hat):
    for i in range(x.size(0)):
        img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28,28)
        plt.imshow(img, cmap='gray')
        plt.show()
        print("Predict:", float(torch.argmax(y_hat[i], dim=-1)))

def test(model, x, y, plot=True):
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
        correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
        total_cnt = float(x.size(0))
        accuracy = correct_cnt / total_cnt
        print("Accuracy: %.4f" % accuracy)
        if plot == True:
            _plot(x, y_hat)

if __name__ == '__main__':
    config = define_argparser()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_config, state_dict = load(config.model_fn, device)
    model = get_model(train_config).to(device)
    model.load_state_dict(state_dict)
    print(model)
    x, y = load_mnist(is_train=False, flatten=True if train_config.model == 'fc' else False)
    x, y = x.to(device), y.to(device)
    test(model, x[:20], y[:20], config.plot)