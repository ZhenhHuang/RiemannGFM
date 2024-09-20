import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from modules import *
from utils.train_utils import EarlyStopping, act_fn
from utils.evall_utils import cal_accuracy, cal_AUC_AP
from utils.logger import create_logger


class Pretrain:
    def __init__(self, configs):
        super(Pretrain, self).__init__()
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.build_model()

    def load_data(self):
        dataset = None
        dataloader = None
        return dataset, dataloader

    def build_model(self):
        model = GeoGFM(n_layers=self.configs.n_layers, in_dim=self.configs.in_dim,
                        out_dim=self.configs.out_dim, bias=self.configs.bias,
                        dropout=self.configs.dropout, activation=act_fn(self.configs.activation))
        self.model = model

    def train(self):
        early_stop = EarlyStopping(self.configs.patience)
        logger = create_logger(self.configs.log_path)
        dataset, dataloader = self.load_data()
        optimizer = Adam(self.model.parameters(), lr=self.configs.lr, weight_decay=self.configs.weight_decay)
        criterion = None    # TODO: self-supervised loss
        for epoch in range(self.configs.pretrain_epoch):
            epoch_loss = []
            for data in dataloader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            train_loss = np.mean(epoch_loss)
            logger.info(f"Epoch {epoch}: train_loss={train_loss}")
            early_stop(train_loss, self.model, self.configs.pretrained_model_path)