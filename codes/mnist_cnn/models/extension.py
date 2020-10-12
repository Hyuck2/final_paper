import torch
import mnist_cnn_cpp_cuda # cnn with cpp and cuda

class CNN_cpp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_00 = torch.nn.Conv2d(1, 32, (3, 3), padding=1)
        self.layer_00.weight = torch.nn.Parameter(torch.zeros(1,0,32))
        print(list(self.parameters()))
        '''
        self.layer = torch.nn.Sequential(
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
            torch.nn.BatchNorm2d(512),
            #torch.nn.squeeze(),
            torch.nn.Linear(512, 50),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(50),
            torch.nn.Linear(50, 10), # MNIST--> outputsize = 10
            torch.nn.Softmax(dim=-1),)
        '''

    def forward(self, x):
        output = self.layer_00(x)
        print(output)
        #return mnist_cnn_cpp_cuda.forward(x)
    '''
    def backward(self, loss):
        return mnist_cnn_cpp_cuda.backward(loss, self.weight, self.bias)
    '''

class CNN_cu(torch.nn.Module):
    def __init__(self):
        super().__init__()