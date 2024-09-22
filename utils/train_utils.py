import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def act_fn(act_str: str):
    if act_str == 'relu':
        return F.relu
    elif act_str == 'leaky_relu':
        return lambda x: F.leaky_relu(x, negative_slope=0.2, inplace=True)
    elif act_str == 'tanh':
        return F.tanh
    elif act_str == 'elu':
        return F.elu
    elif str is None:
        return lambda x: nn.Identity(x)
    else:
        raise NotImplementedError


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, save=True):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if save:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists('./checkpoints'):
           os.mkdir('./checkpoints')
        torch.save(model.state_dict(), f"./checkpoints/{path}")
        self.val_loss_min = val_loss