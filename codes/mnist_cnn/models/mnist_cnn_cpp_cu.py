import torch
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

class CNN_cpp(torch.nn.module):
    pass

class CNN_cu(torch.nn.module):
    pass