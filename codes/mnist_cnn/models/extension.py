import torch
import mnist_cnn_cpp_cuda # cnn with cpp and cuda

class CNN_cpp_functions(torch.autograd.Function):
    @staticmethod
    def forward():
        mnist_cnn_cpp_cuda.forward() # cpp
    
    
    @staticmethod
    def backward():
        mnist_cnn_cpp_cuda.backward() # cpp

class CNN_cu_functions(torch.autograd.Function):
    @staticmethod
    def forward():
        mnist_cnn_cpp_cuda.cu_forward() # cuda
    
    @staticmethod
    def backward():
        mnist_cnn_cpp_cuda.cu_backward() # cuda

class CNN_cpp(torch.nn.Module):
    def __init__(self, output_size):
        self.output_size = output_size
        super().__init__()
        self.blocks = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 32, (3, 3), stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 64, (3, 3), stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 128, (3, 3), stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Conv2d(128, 256, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 256, (3, 3), stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.Conv2d(256, 512, (3, 3), padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Conv2d(512, 512, (3, 3), stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(512, 50),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(50),
            torch.nn.Linear(50, output_size), # MNIST--> outputsize = 10
            torch.nn.Softmax(dim=-1),)
    
    def forward(self, x):
        return CNN_cpp_functions.forward()
    
    def forward(self, x):
        return CNN_cpp_functions.backward()

class CNN_cu(torch.nn.Module):
    pass