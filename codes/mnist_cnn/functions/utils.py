import torch
import matplotlib
from torch.utils.data import Dataset, DataLoader

class MnistDataset(Dataset):
    def __init__(self, data, labels, flatten=True):
        self.data = data
        self.labels = labels
        self.flatten = flatten
        super().__init__()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.flatten:
            x = x.view(-1)
        return x, y

def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms
    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )
    x = dataset.data.float() / 255.
    y = dataset.targets
    if flatten:
        x = x.view(x.size(0), -1)
    return x, y

def loader(config):
    x, y = load_mnist(is_train=True, flatten=False)
    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt
    flatten = True if config.model == 'fc' else False

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    train_x, valid_x = torch.index_select(
        x,
        dim=0,
        index=indices).split([train_cnt, valid_cnt], dim=0)
    train_y, valid_y = torch.index_select(
        y,
        dim=0,
        index=indices).split([train_cnt, valid_cnt], dim=0)
    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=True,)
    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=True,)

    test_x, test_y = load_mnist(is_train=False, flatten=False)
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=False,
    )
    return train_loader, valid_loader, test_loader

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

def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    try:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)
    return total_norm


def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0
    try:
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)
    return total_norm