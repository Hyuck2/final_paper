from copy import deepcopy
import time
import torch
import numpy
from tqdm import tqdm

class trainer():
    def __init__(self, model, config, optimizer, crit):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.crit = crit

    def _train(self, x, y):
        self.model.train()
        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices).split(self.config.batch_size, dim=0)
        y = torch.index_select(y, dim=0, index=indices).split(self.config.batch_size, dim=0)
        total_loss = 0
        forward = 0
        backward = 0
        for i, (x_i, y_i) in tqdm(enumerate(zip(x, y))):
            # forward time check
            start = time.time()
            y_hat_i = self.model(x_i, list(self.model.parameters()))        
            forward += time.time() - start
            # backward time check
            start = time.time()
            loss_i = self.crit(y_hat_i, y_i.squeeze())
            self.optimizer.zero_grad()
            loss_i.backward()
            self.optimizer.step()
            backward += time.time() - start
            if self.config.verbose:
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))
            total_loss += float(loss_i)
        return total_loss / len(x), forward, backward

    def _validate(self, x, y):
        self.model.eval()
        with torch.no_grad():
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices).split(self.config.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indices).split(self.config.batch_size, dim=0)
            total_loss = 0
            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i, list(self.model.parameters()))
                loss_i = self.crit(y_hat_i, y_i.squeeze())
                if self.config.verbose:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))
                total_loss += float(loss_i)
            return total_loss / len(x)

    def train(self, train_data, valid_data):
        lowest_loss = numpy.inf
        best_model = None
        elasped_time = time.time()
        forward = 0
        backward = 0
        optim = 0
        valid = 0
        for epoch in range(self.config.n_epoch):
            start = time.time()
            train_loss, f, b= self._train(train_data[0], train_data[1])
            forward += f
            backward += b
            start = time.time()
            valid_loss = self._validate(valid_data[0], valid_data[1])
            valid += time.time() - start
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch + 1,
                self.config.n_epoch,
                train_loss,
                valid_loss,
                lowest_loss)
            )
        elasped_time = time.time() - elasped_time
        print("time spent : " + str(elasped_time))
        print("forward : " + str(forward) + " backward : " + str(backward) + " valid : " + str(valid))
        self.model.load_state_dict(best_model)
        #torch.save(self.model.state_dict(best_model), self.config.model_dict)