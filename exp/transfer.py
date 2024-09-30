import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from modules import *
from utils.train_utils import EarlyStopping, act_fn, train_node2vec
from utils.data_utils import label2node
from utils.logger import create_logger
from data import load_data, input_dim_dict, ExtractNodeLoader
import os
from tqdm import tqdm
from exp.supervised import LinkPrediction
from exp.pretrain import Pretrain


class ZeroShot(Pretrain):
    def __init__(self, configs, load: bool = False):
        super(ZeroShot, self).__init__(configs)
        self.configs = configs
        self.load = load
        # self.build_model()
        if load:
            pretrained_dict = torch.load(self.configs.pretrained_model_path_ZSL)
            model_dict = self.model.state_dict()
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

    def build_model(self):
        super().build_model()

    def load_data(self, task_level, data_name):
        if task_level == 'node':
            dataset = load_data(root=self.configs.root_path, data_name=data_name)
            data = dataset[0]
            dataloader = ExtractNodeLoader(data, batch_size=self.configs.batch_size,
                                           num_neighbors=self.configs.num_neighbors,
                                           capacity=self.configs.capacity, K_shot=0, num_classes=dataset.num_classes)
        else:
            raise NotImplementedError
        data = label2node(data.clone(), dataset.num_classes)
        return data, dataloader

    def load_transfer_data(self):
        dataset = load_data(root=self.configs.root_path, data_name=self.configs.dataset)
        data = dataset[0]
        transfer_loader = ExtractNodeLoader(data, input_nodes=data.train_mask, batch_size=self.configs.batch_size,
                                        num_neighbors=self.configs.num_neighbors,
                                        capacity=self.configs.capacity, K_shot=0, num_classes=dataset.num_classes)
        return dataset, transfer_loader

    def test(self):
        if self.load is False:
            self.pretrain()
        transfer_set, transfer_loader = self.load_transfer_data()
        transfer_data = label2node(transfer_set[0], transfer_set.num_classes)
        tokens = train_node2vec(transfer_data, self.configs.embed_dim, self.device)
        self.logger.info(f"-----------Zero Shot testing on dataset {self.configs.dataset}-----------")
        total = 0
        matches = 0
        self.model.eval()
        for data in tqdm(transfer_loader):
            data = data.to(self.device)
            data.tokens = tokens(data.n_id)
            x_E, x_H, x_S = self.model(data)
            manifold_H = self.model.manifold_H
            manifold_S = self.model.manifold_S
            x_h = manifold_H.logmap0(x_H)
            x_s = manifold_S.logmap0(x_S)
            x = torch.concat([x_E, x_h, x_s], dim=-1)
            node, label = x[:data.batch_size], x[-transfer_set.num_classes:]
            sim = F.cosine_similarity(node.unsqueeze(1), label.unsqueeze(0), dim=-1)
            correct = (sim.argmax(dim=-1) == data.y[:data.batch_size]).sum()
            matches += correct
            total += data.batch_size
        test_acc = (matches / total).item()
        self.logger.info(f"test_acc={test_acc * 100: .2f}%")
        return test_acc