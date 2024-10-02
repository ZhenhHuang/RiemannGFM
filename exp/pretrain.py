import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from modules import *
from utils.train_utils import EarlyStopping, act_fn, train_node2vec
from utils.logger import create_logger
from data import load_data, input_dim_dict, ExtractNodeLoader
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
        self.build_model()
        # self.data_name = self.configs.dataset

    def load_data(self, task_level, data_name):
        if task_level == 'node':
            dataset = load_data(root=self.configs.root_path, data_name=data_name)
            data = dataset[0]
            dataloader = ExtractNodeLoader(data, batch_size=self.configs.batch_size,
                                           num_neighbors=self.configs.num_neighbors,
                                           capacity=self.configs.capacity)
        else:
            raise NotImplementedError
        return data, dataloader

    def build_model(self):
        model = GeoGFM(n_layers=self.configs.n_layers, in_dim=self.configs.embed_dim,
                      hidden_dim=self.configs.hidden_dim, embed_dim=self.configs.embed_dim,
                      bias=self.configs.bias,
                      dropout=self.configs.dropout,
                       activation=act_fn(self.configs.activation)).to(self.device)
        self.model = model

    def _train(self, load=False, data_name=None):
        if load:
            load_path = os.path.join(self.configs.checkpoints, self.configs.pretrained_model_path) + f".pt"
            self.logger.info(f"---------------Loading pretrained models from {load_path}-------------")
            pretrained_dict = torch.load(load_path)
            model_dict = self.model.state_dict()
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

        path = os.path.join(self.configs.checkpoints, self.configs.pretrained_model_path)
        data, dataloader = self.load_data(self.configs.pretrain_level, data_name)

        tokens = train_node2vec(data, self.configs.embed_dim, self.device)

        optimizer = Adam(self.model.parameters(), lr=self.configs.lr, weight_decay=self.configs.weight_decay)
        for epoch in range(self.configs.pretrain_epochs):
            epoch_loss = []
            for data in tqdm(dataloader):
                optimizer.zero_grad()
                data = data.to(self.device)
                data.tokens = tokens(data.n_id)
                output = self.model(data)
                loss = self.model.loss(output, data)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            train_loss = np.mean(epoch_loss)
            self.logger.info(f"Epoch {epoch}: train_loss={train_loss}")
            self.logger.info(f"---------------Saving pretrained models to {path}_{epoch}.pt-------------")
            torch.save(self.model.state_dict(), path + f"_{epoch}.pt")  # save epoch every
            self.logger.info(f"---------------Saving pretrained models to {path}_{epoch}.pt-------------")
            torch.save(self.model.state_dict(), path + f".pt")  # save for next training

    def pretrain(self):
        if not isinstance(self.configs.pretrain_dataset, list):
            self.configs.pretrain_dataset = [self.configs.pretrain_dataset]
        for i, data_name in enumerate(self.configs.pretrain_dataset):
            load = True
            # self.data_name = data_name
            self.logger.info(f"----------Pretraining on {data_name}--------------")
            # self.build_model()
            if i == 0:
                load = False
            self._train(load, data_name)
            torch.cuda.empty_cache()