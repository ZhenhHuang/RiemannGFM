import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from modules import GeoGFM, EuclideanEncoder, ManifoldEncoder
from utils.train_utils import EarlyStopping, act_fn
from utils.evall_utils import cal_accuracy
from utils.logger import create_logger


class SupervisedExp:
    def __init__(self, configs):
        self.configs = configs
        if self.task == 'NC':
            self.executor = NodeClassification(configs)

    def load_data(self):
        pass

    def train(self):
        pass

    @property
    def task(self):
        return self.configs.task

    @task.setter
    def task(self, task):
        self.task = task
        self.configs.task = task


class NodeClassification:
    def __init__(self, configs, pretrained_model=None, dataset=None, dataloader=None):
        self.configs = configs
        self.pretrained_model = pretrained_model
        self.dataset = dataset
        self.dataloader = dataloader
        self.nc_model = self.load_model()
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def load_model(self):
        if self.pretrained_model is None:
            pretrained_model = GeoGFM(n_layers=self.configs.n_layers, in_dim=self.configs.in_dim,
                                      out_dim=self.configs.out_dim, bias=self.configs.bias,
                                      dropout=self.configs.dropout, activation=act_fn(self.configs.activation))
            self.pretrained_model = pretrained_model.load_state_dict(torch.load(self.configs.pretrained_model))
        for module in self.pretrained_model.modules():
            if not isinstance(module, [EuclideanEncoder, ManifoldEncoder]):
                for param in module.parameters():
                    param.requires_grad = False
        cls_head = nn.Linear(self.configs.out_dim, self.dataset.num_classes)
        nc_model = nn.Sequential(self.pretrained_model, cls_head)
        return nc_model

    def train(self):
        logger = create_logger(self.configs.log_path)
        self.nc_model.train()
        optimizer = Adam(self.nc_model.parameters(), lr=self.configs.lr, weight_decay=self.configs.weight_decay)
        for epoch in range(self.configs.nc_epochs):
            epoch_loss = []
            epoch_acc = []

            loss, acc = self.train_step(optimizer)

            epoch_loss.append(loss.item())
            epoch_acc.append(acc)
            train_loss = np.mean(epoch_loss)
            train_acc = np.mean(epoch_acc)

            logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc * 100: .2f}%")

            if epoch % self.configs.val_every == 0:
                val_loss, val_acc = self.val()
                logger.info(f"Epoch {epoch}: val_loss={val_loss}, val_acc={val_acc * 100: .2f}%")

    def train_step(self, optimizer):
        loss, acc = 0., 0.
        for data in self.dataloader:
            optimizer.zero_grad()
            out = self.nc_model()
            loss, acc = self.cal_loss(out, data.y, data.train_mask)
            loss.backward()
            optimizer.step()
        return loss.item(), acc

    def val(self):
        self.nc_model.eval()
        val_loss = []
        val_acc = []
        with torch.no_grad():
            for data in self.dataloader:
                out = self.nc_model()
                loss, acc = self.cal_loss(out, data.y, data.val_mask)
                val_loss.append(loss.item())
                val_acc.append(acc)
        self.nc_model.train()
        return np.mean(val_loss), np.mean(val_acc)

    def test(self):
        self.nc_model.eval()
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for data in self.dataloader:
                out = self.nc_model()
                loss, acc = self.cal_loss(out, data.y, data.test_mask)
                test_loss.append(loss.item())
                test_acc.append(acc)
        return np.mean(test_loss), np.mean(test_acc)

    def cal_loss(self, output, label, mask):
        out = output[mask]
        y = label[mask]
        loss = F.cross_entropy(out, y)
        acc = cal_accuracy(out, y)
        return loss, acc