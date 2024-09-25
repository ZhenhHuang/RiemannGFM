import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from modules import *
from utils.train_utils import EarlyStopping, act_fn
from utils.evall_utils import cal_accuracy, cal_AUC_AP
from utils.logger import create_logger
from data import load_data, input_dim_dict, ExtractLoader
import os
from tqdm import tqdm


class Pretrain:
    def __init__(self, configs):
        super(Pretrain, self).__init__()
        self.configs = configs
        self.logger = create_logger(self.configs.log_dir + self.configs.log_name)
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # self.build_model()
        self.data_name = None

    def load_data(self, task_level):
        if task_level == 'node':
            dataset = load_data(root=self.configs.root_path, data_name=self.data_name)
            dataloader = ExtractLoader(dataset[0], batch_size=self.configs.batch_size,
                                       num_neighbors=self.configs.num_neighbors)
        else:
            raise NotImplementedError
        return dataloader

    def build_model(self):
        model = GeoGFM(n_layers=self.configs.n_layers, in_dim=input_dim_dict[self.data_name],
                      hidden_dim=self.configs.hidden_dim, embed_dim=self.configs.embed_dim,
                      bias=self.configs.bias,
                      dropout=self.configs.dropout,
                       activation=act_fn(self.configs.activation)).to(self.device)
        self.model = model

    def _train(self, load=False):
        path = os.path.join(self.configs.checkpoints, self.configs.pretrained_model_path)
        if load:
            self.logger.info(f"---------------Loading pretrained models from {path}-------------")
            pretrained_dict = torch.load(path)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'init_block' not in k}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        # early_stop = EarlyStopping(self.configs.patience)
        dataloader = self.load_data(self.configs.pretrain_level)
        optimizer = Adam(self.model.parameters(), lr=self.configs.lr, weight_decay=self.configs.weight_decay)
        for epoch in range(self.configs.pretrain_epochs):
            epoch_loss = []
            for data in tqdm(dataloader):
                optimizer.zero_grad()
                data = data.to(self.device)
                output = self.model(data)
                loss = self.model.loss(output, data)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            train_loss = np.mean(epoch_loss)
            self.logger.info(f"Epoch {epoch}: train_loss={train_loss}")
            self.logger.info(f"---------------Saving pretrained models to {path}-------------")
            torch.save(self.model.state_dict(), path)
            # early_stop(train_loss, self.model, self.configs.checkpoints, self.configs.pretrained_model_path)
            # if early_stop.early_stop:
            #     print("---------Early stopping--------")
            #     break

    def pretrain(self):
        if not isinstance(self.configs.pretrain_dataset, list):
            self.configs.pretrain_dataset = [self.configs.pretrain_dataset]
        for i, data_name in enumerate(self.configs.pretrain_dataset):
            load = True
            self.data_name = data_name
            self.logger.info(f"----------Pretraining on {data_name}--------------")
            self.build_model()
            if i == 0:
                load = False
            self._train(load)
            torch.cuda.empty_cache()
