import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from modules import *
from utils.train_utils import EarlyStopping, act_fn
from utils.evall_utils import cal_accuracy, cal_AUC_AP
from utils.logger import create_logger


class SupervisedExp:
    def __init__(self, configs, pretrained_model=None):
        self.configs = configs
        self.pretrained_model = pretrained_model
        if self.pretrained_model is None:
            pretrained_model = GeoGFM(n_layers=self.configs.n_layers, in_dim=self.configs.in_dim,
                                      out_dim=self.configs.out_dim, bias=self.configs.bias,
                                      dropout=self.configs.dropout, activation=act_fn(self.configs.activation))
            self.pretrained_model = pretrained_model.load_state_dict(torch.load(self.configs.pretrained_model))
        for module in self.pretrained_model.modules():
            if not isinstance(module, [EuclideanEncoder, ManifoldEncoder]):
                for param in module.parameters():
                    param.requires_grad = False
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.load_data()

    def load_model(self):
        pass

    def load_data(self):
        """
        According to self.configs.dataset and other args
        :return:
        """
        self.dataset = None
        self.dataloader = None

    def train(self):
        pass

    def train_step(self, data, optimizer):
        pass

    def val(self):
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
        cls_head = NodeClsHead(self.configs.out_dim, self.dataset.num_classes)
        nc_model = nn.Sequential(self.pretrained_model, cls_head)
        return nc_model

    def train(self):
        logger = create_logger(self.configs.log_path)
        self.nc_model.train()
        optimizer = Adam(self.nc_model.parameters(), lr=self.configs.lr_nc, weight_decay=self.configs.weight_decay_nc)
        for epoch in range(self.configs.nc_epochs):
            epoch_loss = []
            epoch_acc = []

            for data in self.dataloader:
                loss, acc = self.train_step(data, optimizer)
                epoch_loss.append(loss)
                epoch_acc.append(acc)

            train_loss = np.mean(epoch_loss)
            train_acc = np.mean(epoch_acc)

            logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc * 100: .2f}%")

            if epoch % self.configs.val_every == 0:
                val_loss, val_acc = self.val()
                logger.info(f"Epoch {epoch}: val_loss={val_loss}, val_acc={val_acc * 100: .2f}%")

    def train_step(self, data, optimizer):
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
        test_acc = []
        with torch.no_grad():
            for data in self.dataloader:
                out = self.nc_model()
                loss, acc = self.cal_loss(out, data.y, data.test_mask)
                test_acc.append(acc)
        return np.mean(test_acc)

    def cal_loss(self, output, label, mask):
        out = output[mask]
        y = label[mask]
        loss = F.cross_entropy(out, y)
        acc = cal_accuracy(out, y)
        return loss, acc


class GraphClassification(SupervisedExp):
    def __init__(self, configs, pretrained_model=None):
        super(GraphClassification, self).__init__(configs, pretrained_model)
        self.gc_model = self.load_model()

    def load_model(self):
        cls_head = nn.Linear(self.configs.out_dim, self.dataset.num_classes)
        gc_model = nn.Sequential(self.pretrained_model, cls_head)
        return gc_model

    def train(self):
        logger = create_logger(self.configs.log_path)
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

            logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc * 100: .2f}%")

            if epoch % self.configs.val_every == 0:
                val_loss, val_acc = self.val()
                logger.info(f"Epoch {epoch}: val_loss={val_loss}, val_acc={val_acc * 100: .2f}%")

    def train_step(self, data, optimizer):
        optimizer.zero_grad()
        out = self.gc_model()
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=-1)
        correct = (pred == data.y).sum().item()
        return loss.item(), correct

    def val(self):
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


class LinkPrediction(SupervisedExp):
    def __init__(self, configs, pretrained_model=None):
        super(LinkPrediction, self).__init__(configs, pretrained_model)
        self.lp_model = self.load_model()

    def load_model(self):
        cls_head = LinkPredHead(self.configs.out_dim, self.configs.out_dim_lp, self.configs.r, self.configs.s)
        lp_model = nn.Sequential(self.pretrained_model, cls_head)
        return lp_model

    def train(self):
        logger = create_logger(self.configs.log_path)
        self.lp_model.train()
        optimizer = Adam(self.lp_model.parameters(), lr=self.configs.lr_lp, weight_decay=self.configs.weight_decay_lp)
        for epoch in range(self.configs.lp_epochs):
            epoch_loss = []
            epoch_label = []
            epoch_pred = []

            for data in self.dataloader:
                loss, pred, label = self.train_step(data, optimizer)
                epoch_loss.append(loss)
                epoch_label += label
                epoch_pred += pred

            train_loss = np.mean(epoch_loss)
            train_auc, train_ap = cal_AUC_AP(epoch_pred, epoch_label)

            logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_auc={train_auc * 100: .2f}%, "
                        f"train_ap={train_ap * 100: .2f}%")

            if epoch % self.configs.val_every == 0:
                val_loss, val_auc, val_ap = self.val()
                logger.info(f"Epoch {epoch}: val_loss={val_loss}, val_auc={val_auc * 100: .2f}%, "
                            f"val_ap={val_ap * 100: .2f}%")

    def train_step(self, data, optimizer):
        optimizer.zero_grad()
        out_pos, out_neg = self.lp_model(data.pos_edge_index, data.neg_edge_index)
        loss, pred, label = self.cal_loss(out_pos, out_neg)
        loss.backward()
        optimizer.step()
        return loss.item(), pred, label

    def val(self):
        self.lp_model.eval()
        val_loss = []
        val_label = []
        val_pred = []
        with torch.no_grad():
            for data in self.dataloader:
                out_pos, out_neg = self.lp_model(data.pos_edge_index, data.neg_edge_index)
                loss, pred, label = self.cal_loss(out_pos, out_neg)
                val_loss.append(loss.item())
                val_label += label
                val_pred += pred
        val_loss = np.mean(val_loss)
        val_auc, val_ap = cal_AUC_AP(val_pred, val_label)
        self.lp_model.train()
        return val_loss, val_auc, val_ap

    def test(self):
        self.lp_model.eval()
        test_label = []
        test_pred = []
        with torch.no_grad():
            for data in self.dataloader:
                out_pos, out_neg = self.lp_model(data.pos_edge_index, data.neg_edge_index)
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