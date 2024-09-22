import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from modules import *
from utils.train_utils import EarlyStopping, act_fn
from utils.evall_utils import cal_accuracy, cal_AUC_AP
from utils.logger import create_logger
from data import NodeClsDataset, LinkPredDataset, load_data, input_dim_dict
from torch_geometric.loader import DataLoader


class SupervisedExp:
    def __init__(self, configs, pretrained_model=None):
        self.configs = configs
        self.logger = create_logger(self.configs.log_dir + self.configs.log_name)
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if pretrained_model is None:
            pretrained_model = GeoGFM(n_layers=self.configs.n_layers, in_dim=input_dim_dict[self.configs.dataset],
                                      out_dim=self.configs.embed_dim, bias=self.configs.bias,
                                      dropout=self.configs.dropout, activation=act_fn(self.configs.activation))
            pretrained_model = pretrained_model.load_state_dict(
                torch.load(self.configs.checkpoints + self.configs.pretrained_model_path)
            )
        for module in self.pretrained_model.modules():
            if not isinstance(module, [EuclideanEncoder, ManifoldEncoder]):
                for param in module.parameters():
                    param.requires_grad = False
        self.pretrained_model = pretrained_model.to(self.device)

    def load_model(self):
        pass

    def load_data(self, split):
        pass

    def train(self):
        pass

    def train_step(self, data, optimizer):
        pass

    def val(self, val_loader):
        pass

    def test(self):
        pass

    def cal_loss(self, **kwargs):
        pass


class NodeClassification(SupervisedExp):
    # TODO: Transductive or Inductive
    def __init__(self, configs, pretrained_model=None):
        super(NodeClassification, self).__init__(configs, pretrained_model)
        self.nc_model = self.load_model()

    def load_model(self):
        cls_head = NodeClsHead(self.configs.embed_dim, self.dataset.num_classes).to(self.device)
        nc_model = nn.Sequential(self.pretrained_model, cls_head)
        return nc_model

    def load_data(self, split: str):
        dataset = NodeClsDataset(raw_dataset=load_data(root=self.configs.root_path,
                                                       data_name=self.configs.dataset),
                                 configs=self.configs,
                                 split=split)
        dataloader = DataLoader(dataset, batch_size=1)
        return dataset, dataloader

    def train(self):
        self.nc_model.train()
        optimizer = Adam(self.nc_model.parameters(), lr=self.configs.lr_nc, weight_decay=self.configs.weight_decay_nc)
        train_set, train_loader = self.load_data("train")
        val_set, val_loader = self.load_data("val")
        for epoch in range(self.configs.nc_epochs):
            epoch_loss = []
            epoch_acc = []

            for data in train_loader:
                data = data.to(self.device)
                loss, acc = self.train_step(data, optimizer)
                epoch_loss.append(loss)
                epoch_acc.append(acc)

            train_loss = np.mean(epoch_loss)
            train_acc = np.mean(epoch_acc)

            self.logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc * 100: .2f}%")

            if epoch % self.configs.val_every == 0:
                val_loss, val_acc = self.val(val_loader)
                self.logger.info(f"Epoch {epoch}: val_loss={val_loss}, val_acc={val_acc * 100: .2f}%")

    def train_step(self, data, optimizer):
        optimizer.zero_grad()
        out = self.nc_model(data)
        loss, acc = self.cal_loss(out, data.y, data.train_mask)
        loss.backward()
        optimizer.step()
        return loss.item(), acc

    def val(self, val_loader):
        self.nc_model.eval()
        val_loss = []
        val_acc = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                out = self.nc_model(data)
                loss, acc = self.cal_loss(out, data.y, data.val_mask)
                val_loss.append(loss.item())
                val_acc.append(acc)
        self.nc_model.train()
        return np.mean(val_loss), np.mean(val_acc)

    def test(self):
        test_set, test_loader = self.load_data("test")
        self.nc_model.eval()
        test_acc = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out = self.nc_model(data)
                loss, acc = self.cal_loss(out, data.y, data.test_mask)
                test_acc.append(acc)
        return np.mean(test_acc)

    def cal_loss(self, output, label, mask):
        # TODO: Inductive loss or transductive loss
        out = output[mask]
        y = label[mask]
        loss = F.cross_entropy(out, y)
        acc = cal_accuracy(out, y)
        return loss, acc


class LinkPrediction(SupervisedExp):
    def __init__(self, configs, pretrained_model=None):
        super(LinkPrediction, self).__init__(configs, pretrained_model)
        self.lp_model = self.load_model()

    def load_data(self, split):
        dataset = LinkPredDataset(raw_dataset=load_data(root=self.configs.root_path,
                                                       data_name=self.configs.dataset),
                                 configs=self.configs,
                                 split=split)
        dataloader = DataLoader(dataset, batch_size=1)
        return dataset, dataloader

    def load_model(self):
        cls_head = LinkPredHead(self.configs.embed_dim, self.configs.embed_dim_lp).to(self.device)
        lp_model = nn.Sequential(self.pretrained_model, cls_head)
        return lp_model

    def train(self):
        self.lp_model.train()
        optimizer = Adam(self.lp_model.parameters(), lr=self.configs.lr_lp, weight_decay=self.configs.weight_decay_lp)
        train_set, train_loader = self.load_data("train")
        val_set, val_loader = self.load_data("val")
        for epoch in range(self.configs.lp_epochs):
            epoch_loss = []
            epoch_label = []
            epoch_pred = []

            for data in train_loader:
                data = data.to(self.device)
                loss, pred, label = self.train_step(data, optimizer)
                epoch_loss.append(loss)
                epoch_label += label
                epoch_pred += pred

            train_loss = np.mean(epoch_loss)
            train_auc, train_ap = cal_AUC_AP(epoch_pred, epoch_label)

            self.logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_auc={train_auc * 100: .2f}%, "
                        f"train_ap={train_ap * 100: .2f}%")

            if epoch % self.configs.val_every == 0:
                val_loss, val_auc, val_ap = self.val(val_loader)
                self.logger.info(f"Epoch {epoch}: val_loss={val_loss}, val_auc={val_auc * 100: .2f}%, "
                            f"val_ap={val_ap * 100: .2f}%")

    def train_step(self, data, optimizer):
        optimizer.zero_grad()
        out_pos, out_neg = self.lp_model(data, data.edge_index, data.neg_edge_index)
        loss, pred, label = self.cal_loss(out_pos, out_neg)
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
                out_pos, out_neg = self.lp_model(data, data.edge_index, data.neg_edge_inde)
                loss, pred, label = self.cal_loss(out_pos, out_neg)
                val_loss.append(loss.item())
                val_label += label
                val_pred += pred
        val_loss = np.mean(val_loss)
        val_auc, val_ap = cal_AUC_AP(val_pred, val_label)
        self.lp_model.train()
        return val_loss, val_auc, val_ap

    def test(self):
        test_set, test_loader = self.load_data("test")
        self.lp_model.eval()
        test_label = []
        test_pred = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out_pos, out_neg = self.lp_model(data, data.edge_index, data.neg_edge_inde)
                loss, pred, label = self.cal_loss(out_pos, out_neg)
                test_label += label
                test_pred += pred
        test_auc, test_ap = cal_AUC_AP(test_pred, test_label)
        return test_auc, test_ap

    def cal_loss(self, out_pos, out_neg):
        loss = F.binary_cross_entropy_with_logits(out_pos, torch.ones_like(out_pos)) + \
               F.binary_cross_entropy_with_logits(out_neg, torch.zeros_like(out_neg))
        label = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.detach().cpu().numpy()) + list(neg_scores.detach().cpu().numpy())
        return loss, preds, label


class GraphClassification(SupervisedExp):
    def __init__(self, configs, pretrained_model=None):
        super(GraphClassification, self).__init__(configs, pretrained_model)
        self.gc_model = self.load_model()

    def load_model(self):
        cls_head = nn.Linear(self.configs.out_dim, self.dataset.num_classes)
        gc_model = nn.Sequential(self.pretrained_model, cls_head)
        return gc_model

    def load_data(self, split):
        # TODO
        pass

    def train(self):
        self.gc_model.train()
        optimizer = Adam(self.gc_model.parameters(), lr=self.configs.lr_gc, weight_decay=self.configs.weight_decay_gc)
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

    def train_step(self, data, optimizer):
        optimizer.zero_grad()
        out = self.gc_model()
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
                out = self.gc_model()
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
                out = self.gc_model()
                pred = out.argmax(dim=-1)
                test_correct += (pred == data.y).sum().item()
        return test_correct / len(self.dataloader.dataset)

    def cal_loss(self, output, label):
        loss = F.cross_entropy(output, label)
        acc = cal_accuracy(output, label)
        return loss, acc