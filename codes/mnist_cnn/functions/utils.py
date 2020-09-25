import torch
import matplotlib
from torch.utils.data import Dataset, DataLoader

class loader:
    def __init__(self):
        pass

class logger:
    def __init__(self, parameters):
        path = './log/'
        for parameter in parameters:
            path = path + str(parameter) + '_'
        path = path + '.log'
        self.path = path
        self.log = open(path, 'a')

    def write(self, data):
        self.log.write()

    def cvt_to_csv(self):
        csv = self.log
        return csv

    def close(self):
        self.log.close()

    def plot(self, config):
        pass