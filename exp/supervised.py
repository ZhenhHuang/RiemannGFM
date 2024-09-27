import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from geoopt.optim import RiemannianAdam
import numpy as np
from modules import *
from utils.train_utils import EarlyStopping, act_fn
from utils.evall_utils import cal_accuracy, cal_AUC_AP
from utils.logger import create_logger
from data import load_data, ExtractNodeLoader, ExtractLinkLoader, input_dim_dict, class_num_dict
from torch_geometric.transforms import RandomLinkSplit
import os
from tqdm import tqdm


class SupervisedExp:
    def __init__(self, configs, pretrained_model=None, load=False, finetune=False):
        self.configs = configs
        self.logger = create_logger(self.configs.log_dir + self.configs.log_name)
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if pretrained_model is None:
            pretrained_model = GeoGFM(n_layers=self.configs.n_layers, in_dim=input_dim_dict[self.configs.dataset],
                                      hidden_dim=self.configs.hidden_dim, embed_dim=self.configs.embed_dim,
                                      bias=self.configs.bias,
                                      dropout=self.configs.dropout, activation=act_fn(self.configs.activation))
            if load:
                path = os.path.join(self.configs.checkpoints, self.configs.pretrained_model_path) + f"_{self.configs.id}.pt"
                self.logger.info(f"---------------Loading pretrained models from {path}-------------")

                pretrained_dict = torch.load(path)
                model_dict = pretrained_model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'Euc_init' not in k}
                model_dict.update(pretrained_dict)
                pretrained_model.load_state_dict(model_dict)

        if finetune:
            self.logger.info("----------Freezing weights-----------")
            for module in pretrained_model.modules():
                if not isinstance(module, (EuclideanEncoder, ManifoldEncoder)):
                    for param in module.parameters():
                        param.requires_grad = False
        self.pretrained_model = pretrained_model.to(self.device)

    def load_model(self):
        raise NotImplementedError

    def load_data(self, split):
        raise NotImplementedError

    def train(self):
        pass

    def train_step(self, data, optimizer):
        pass

    def val(self, val_loader):
        pass

    def test(self, test_loader):
        pass

    def cal_loss(self, **kwargs):
        raise NotImplementedError


class NodeClassification(SupervisedExp):
    def __init__(self, configs, pretrained_model=None, load=False, finetune=False):
        super(NodeClassification, self).__init__(configs, pretrained_model, load, finetune)
        self.nc_model = self.load_model()

    def load_model(self):
        nc_model = NodeClsHead(self.pretrained_model, 3 * self.configs.embed_dim,
                               class_num_dict[self.configs.dataset]).to(self.device)
        return nc_model

    def load_data(self, split: str):
        dataset = load_data(root=self.configs.root_path, data_name=self.configs.dataset)
        dataloader = ExtractNodeLoader(dataset[0], batch_size=self.configs.batch_size,
                                   num_neighbors=self.configs.num_neighbors,
                                   capacity=self.configs.capacity)
        return dataloader

    def train(self):
        self.nc_model.train()
        optimizer = RiemannianAdam(self.nc_model.parameters(), lr=self.configs.lr_nc,
                         weight_decay=self.configs.weight_decay_nc)
        dataloader = self.load_data("train")
        early_stop = EarlyStopping(self.configs.patience)
        for epoch in range(self.configs.nc_epochs):
            epoch_loss = []
            total = 0
            matches = 0

            for data in tqdm(dataloader):
                data = data.to(self.device)
                loss, correct, num = self.train_step(data, optimizer)
                epoch_loss.append(loss)
                matches += correct
                total += num

            train_loss = np.mean(epoch_loss)
            train_acc = (matches / total).item()

            self.logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc * 100: .2f}%")

            if epoch % self.configs.val_every == 0:
                val_loss, val_acc = self.val(dataloader)
                self.logger.info(f"Epoch {epoch}: val_loss={val_loss}, val_acc={val_acc * 100: .2f}%")
                early_stop(val_loss, self.nc_model, self.configs.checkpoints, self.configs.task_model_path)
                if early_stop.early_stop:
                    print("---------Early stopping--------")
                    break
        test_acc = self.test(dataloader)
        self.logger.info(f"test_acc={test_acc * 100: .2f}%")

    def train_step(self, data, optimizer):
        optimizer.zero_grad()
        out = self.nc_model(data)
        loss, correct = self.cal_loss(out, data.y, data.train_mask)
        loss.backward()
        optimizer.step()
        return loss.item(), correct, data.train_mask.sum()

    def val(self, val_loader):
        self.nc_model.eval()
        val_loss = []
        total = 0
        matches = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                out = self.nc_model(data)
                loss, correct = self.cal_loss(out, data.y, data.val_mask)
                val_loss.append(loss.item())
                matches += correct
                total += data.val_mask.sum()
        self.nc_model.train()
        return np.mean(val_loss), (matches / total).item()

    def test(self, test_loader=None):
        test_loader = self.load_data("test") if test_loader is None else test_loader
        self.nc_model.eval()
        self.logger.info("--------------Testing--------------------")
        path = os.path.join(self.configs.checkpoints, self.configs.task_model_path)
        self.logger.info(f"--------------Loading from {path}--------------------")
        self.nc_model.load_state_dict(torch.load(path))
        total = 0
        matches = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out = self.nc_model(data)
                loss, correct = self.cal_loss(out, data.y, data.test_mask)
                matches += correct
                total += data.test_mask.sum()
        test_acc = (matches / total).item()
        self.logger.info(f"test_acc={test_acc * 100: .2f}%")
        return test_acc

    def cal_loss(self, output, label, mask):
        out = output if self.configs.nc_mode == 'inductive' else output[mask]
        y = label if self.configs.nc_mode == 'inductive' else label[mask]
        loss = F.cross_entropy(out, y)
        correct = (out.argmax(dim=-1) == y).sum()
        return loss, correct


class LinkPrediction(SupervisedExp):
    def __init__(self, configs, pretrained_model=None, load=False, finetune=False):
        super(LinkPrediction, self).__init__(configs, pretrained_model, load, finetune)
        self.lp_model = self.load_model()

    def load_data(self, split):
        dataset = load_data(root=self.configs.root_path, data_name=self.configs.dataset)
        train_data, val_data, test_data = RandomLinkSplit(is_undirected=True,
                                                          add_negative_train_samples=False)(dataset[0])
        train_loader = ExtractLinkLoader(train_data, batch_size=self.configs.batch_size,
                                   num_neighbors=self.configs.num_neighbors,
                                         neg_sampling_ratio=2.0,
                                   capacity=self.configs.capacity)
        val_loader = ExtractLinkLoader(val_data, batch_size=self.configs.batch_size,
                                     num_neighbors=self.configs.num_neighbors,
                                       neg_sampling_ratio=2.0,
                                     capacity=self.configs.capacity)
        test_loader = ExtractLinkLoader(test_data, batch_size=self.configs.batch_size,
                                     num_neighbors=self.configs.num_neighbors,
                                        neg_sampling_ratio=2.0,
                                     capacity=self.configs.capacity)
        if split == 'test':
            return test_loader
        return train_loader, val_loader, test_loader

    def load_model(self):
        lp_model = LinkPredHead(self.pretrained_model,
                                3 * self.configs.embed_dim,
                                self.configs.embed_dim_lp).to(self.device)
        return lp_model

    def train(self):
        self.lp_model.train()
        optimizer = Adam(self.lp_model.parameters(), lr=self.configs.lr_lp, weight_decay=self.configs.weight_decay_lp)
        train_loader, val_loader, test_loader = self.load_data(None)
        early_stop = EarlyStopping(self.configs.patience)
        for epoch in range(self.configs.lp_epochs):
            epoch_loss = []
            epoch_label = []
            epoch_pred = []

            for data in tqdm(train_loader):
                data = data.to(self.device)
                loss, pred, label = self.train_step(data, optimizer)
                epoch_loss.append(loss)
                epoch_label.append(label)
                epoch_pred.append(pred)

            train_loss = np.mean(epoch_loss)
            epoch_pred = torch.cat(epoch_pred, dim=-1).detach().cpu().numpy()
            epoch_label = torch.cat(epoch_label, dim=-1).cpu().numpy()
            train_auc, train_ap = cal_AUC_AP(epoch_pred, epoch_label)

            self.logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_auc={train_auc * 100: .2f}%, "
                        f"train_ap={train_ap * 100: .2f}%")

            if epoch % self.configs.val_every == 0:
                val_loss, val_auc, val_ap = self.val(val_loader)
                self.logger.info(f"Epoch {epoch}: val_loss={val_loss}, val_auc={val_auc * 100: .2f}%, "
                            f"val_ap={val_ap * 100: .2f}%")
                early_stop(val_loss, self.lp_model, self.configs.checkpoints, self.configs.task_model_path)
                if early_stop.early_stop:
                    print("---------Early stopping--------")
                    break
        test_auc, test_ap = self.test(test_loader)
        self.logger.info(f"test_auc={test_auc * 100: .2f}%, "
                         f"test_ap={test_ap * 100: .2f}%")

    def train_step(self, data, optimizer):
        optimizer.zero_grad()
        pred, label = self.lp_model(data)
        loss = F.binary_cross_entropy_with_logits(pred, label)
        loss.backward()
        optimizer.step()
        return loss.item(), pred, label

    def val(self, val_loader):
        self.lp_model.eval()
        val_loss = []
        val_label = []
        val_pred = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                pred, label = self.lp_model(data)
                loss = F.binary_cross_entropy_with_logits(pred, label)
                val_loss.append(loss.item())
                val_label.append(label)
                val_pred.append(pred)
        val_loss = np.mean(val_loss)
        val_pred = torch.cat(val_pred, dim=-1).detach().cpu().numpy()
        val_label = torch.cat(val_label, dim=-1).cpu().numpy()
        val_auc, val_ap = cal_AUC_AP(val_pred, val_label)
        self.lp_model.train()
        return val_loss, val_auc, val_ap

    def test(self, test_loader):
        test_loader = self.load_data("test") if test_loader is None else test_loader
        self.lp_model.eval()
        self.logger.info("--------------Testing--------------------")
        path = os.path.join(self.configs.checkpoints, self.configs.task_model_path)
        self.logger.info(f"--------------Loading from {path}--------------------")
        self.lp_model.load_state_dict(torch.load(path))
        test_label = []
        test_pred = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                pred, label = self.lp_model(data)
                test_label.append(label)
                test_pred.append(pred)
            test_pred = torch.cat(test_pred, dim=-1).detach().cpu().numpy()
            test_label = torch.cat(test_label, dim=-1).cpu().numpy()
            test_auc, test_ap = cal_AUC_AP(test_pred, test_label)
        return test_auc, test_ap


class GraphClassification(SupervisedExp):
    def __init__(self, configs, pretrained_model=None, load=False, finetune=False):
        super(GraphClassification, self).__init__(configs, pretrained_model, load, finetune)
        self.gc_model = self.load_model()

    def load_model(self):
        gc_model = nn.Linear(self.pretrained_model, 3 * self.configs.embed_dim,
                             class_num_dict[self.configs.dataset]).to(self.device)
        return gc_model

    def load_data(self, split):
        # TODO
        pass

    def train(self):
        self.gc_model.train()
        optimizer = Adam(self.gc_model.parameters(), lr=self.configs.lr_gc,
                         weight_decay=self.configs.weight_decay_gc)
        early_stop = EarlyStopping(self.configs.patience)
        for epoch in range(self.configs.gc_epochs):
            epoch_loss = []
            epoch_correct = 0.

            for data in self.dataloader:
                loss, correct = self.train_step(data, optimizer)
                epoch_correct += correct
                epoch_loss.append(epoch_loss)

            train_acc = epoch_correct / len(self.dataloader.dataset)
            train_loss = np.mean(epoch_loss)

            self.logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc * 100: .2f}%")

            if epoch % self.configs.val_every == 0:
                val_loss, val_acc = self.val()
                self.logger.info(f"Epoch {epoch}: val_loss={val_loss}, val_acc={val_acc * 100: .2f}%")
                early_stop(val_loss, self.nc_model, self.configs.checkpoints, self.configs.task_model_path)
                if early_stop.early_stop:
                    print("---------Early stopping--------")
                    break
        test_acc = self.test()
        self.logger.info(f"test_acc={test_acc * 100: .2f}%")

    def train_step(self, data, optimizer):
        optimizer.zero_grad()
        out = self.gc_model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=-1)
        correct = (pred == data.y).sum().item()
        return loss.item(), correct

    def val(self, val_loader):
        self.gc_model.eval()
        val_loss = []
        val_correct = 0.
        with torch.no_grad():
            for data in self.dataloader:
                out = self.gc_model(data)
                loss = F.cross_entropy(out, data.y)
                val_loss.append(loss.item())
                pred = out.argmax(dim=-1)
                val_correct += (pred == data.y).sum().item()
        self.gc_model.train()
        return np.mean(val_loss), val_correct / len(self.dataloader.dataset)

    def test(self):
        self.gc_model.eval()
        test_correct = 0.
        with torch.no_grad():
            for data in self.dataloader:
                out = self.gc_model(data)
                pred = out.argmax(dim=-1)
                test_correct += (pred == data.y).sum().item()
        return test_correct / len(self.dataloader.dataset)

    def cal_loss(self, output, label):
        loss = F.cross_entropy(output, label)
        acc = cal_accuracy(output, label)
        return loss, acc