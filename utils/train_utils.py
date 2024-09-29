import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import Node2Vec
from torch.optim import SparseAdam
import warnings
warnings.filterwarnings("ignore")


def train_node2vec(data, embed_dim, device):
    model = Node2Vec(
        data.edge_index,
        embedding_dim=embed_dim,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        sparse=True,
    ).to(device)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = SparseAdam(model.parameters(), lr=0.01)
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss = total_loss / len(loader)
        model.eval()
        with torch.no_grad():
            z = model()
            acc = model.test(
                train_z=z[data.train_mask],
                train_y=data.y[data.train_mask],
                test_z=z[data.test_mask],
                test_y=data.y[data.test_mask],
                max_iter=150,
            )
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
    return model


def act_fn(act_str: str):
    if act_str == 'relu':
        return F.relu
    elif act_str == 'leaky_relu':
        return lambda x: F.leaky_relu(x, negative_slope=0.2, inplace=True)
    elif act_str == 'tanh':
        return F.tanh
    elif act_str == 'elu':
        return F.elu
    elif act_str is None:
        return lambda x: nn.Identity()(x)
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

    def __call__(self, val_loss, model, dir, file, save=True):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if save:
                self.save_checkpoint(val_loss, model, dir, file)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if save:
                self.save_checkpoint(val_loss, model, dir, file)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, dir, file):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if not os.path.exists(dir):
            os.mkdir(dir)
        torch.save(model.state_dict(), os.path.join(dir, file))
        self.val_loss_min = val_loss
