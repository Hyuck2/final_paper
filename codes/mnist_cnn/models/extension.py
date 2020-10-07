import torch
import mnist_cnn_cpp_cuda # cnn with cpp and cuda

class CNN_cpp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mnist_cnn_cpp_cuda.forward(x)
    
    def backward(self, loss):
        return mnist_cnn_cpp_cuda.backward(loss)

class CNN_cu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mnist_cnn_cpp_cuda.cu_forward(x)
    
    def backward(self, loss):
        return mnist_cnn_cpp_cuda.cu_backward(loss)
