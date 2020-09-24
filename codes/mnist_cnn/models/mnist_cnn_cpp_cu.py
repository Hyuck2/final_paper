import torch
import torch.nn as nn
import mnist_cnn # cnn with cpp and cuda

class CNN_cpp_functions(torch.autograd.Function):
    @staticmethod
    def forward():
        pass
    
    
    @staticmethod
    def backward():
        pass

class CNN_cu_functions(torch.autograd.Function):
    @staticmethod
    def forward():
        pass

    
    
    @staticmethod
    def backward():
        pass

class CNN_cpp(nn.module):
    pass

class CNN_cu(nn.module):
    pass