import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from modules import *
from utils.train_utils import EarlyStopping, act_fn
from utils.evall_utils import cal_accuracy, cal_AUC_AP
from utils.logger import create_logger
from data import load_data, PretrainingNodeDataset, input_dim_dict
from torch_geometric.loader import DataLoader


class Pretrain:
    def __init__(self, configs):
        super(Pretrain, self).__init__()
        self.configs = configs
        self.logger = create_logger(self.configs.log_dir + self.configs.log_name)
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.build_model()

    def load_data(self, task_level):
        if task_level == 'node':
            dataset = PretrainingNodeDataset(load_data(root=self.configs.root_path,
                                                       data_name=self.configs.dataset),
                                             self.configs)
            dataloader = DataLoader(dataset, batch_size=1)
        else:
            raise NotImplementedError
        return dataset, dataloader

    def build_model(self):
        model = GeoGFM(n_layers=self.configs.n_layers, in_dim=input_dim_dict[self.configs.dataset],
                        out_dim=self.configs.embed_dim, bias=self.configs.bias,
                        dropout=self.configs.dropout, activation=act_fn(self.configs.activation)).to(self.device)
        self.model = model

    def train(self, load=False):
        if load:
            self.model.load_state_dict(
                torch.load(self.configs.checkpoints + self.configs.pretrained_model_path)
            )
        early_stop = EarlyStopping(self.configs.patience)
        dataset, dataloader = self.load_data(self.configs.pretrain_level)
        optimizer = Adam(self.model.parameters(), lr=self.configs.lr, weight_decay=self.configs.weight_decay)
        criterion = None    # TODO: self-supervised loss
        for epoch in range(self.configs.pretrain_epoch):
            epoch_loss = []
            for data in dataloader:
                data = data.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            train_loss = np.mean(epoch_loss)
            self.logger.info(f"Epoch {epoch}: train_loss={train_loss}")
            early_stop(train_loss, self.model, self.configs.checkpoints + self.configs.pretrained_model_path)