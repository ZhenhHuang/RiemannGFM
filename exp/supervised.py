import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from modules import *
from utils.train_utils import EarlyStopping, act_fn
from utils.evall_utils import cal_accuracy, cal_AUC_AP, cal_F1
from utils.logger import create_logger
from data import load_data, ExtractNodeLoader, ExtractLinkLoader, input_dim_dict, class_num_dict, get_eigen_tokens
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
            pretrained_model = GeoGFM(n_layers=self.configs.n_layers, in_dim=self.configs.embed_dim,
                                      hidden_dim=self.configs.hidden_dim, embed_dim=self.configs.embed_dim,
                                      bias=self.configs.bias,
                                      dropout=self.configs.dropout, activation=act_fn(self.configs.activation))
            if load:
                path = os.path.join(self.configs.checkpoints, self.configs.pretrained_model_path) + ".pt"
                self.logger.info(f"---------------Loading pretrained models from {path}-------------")

                pretrained_dict = torch.load(path)
                model_dict = pretrained_model.state_dict()
                model_dict.update(pretrained_dict)
                pretrained_model.load_state_dict(model_dict)

        if finetune:
            self.logger.info("----------Freezing weights-----------")
            for module in pretrained_model.modules():
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
        nc_model = NodeClsHead(self.pretrained_model, 2 * self.configs.embed_dim + input_dim_dict[self.configs.dataset],
                               self.configs.nc_hidden_dim,
                               class_num_dict[self.configs.dataset]).to(self.device)
        return nc_model

    def load_data(self, split: str):
        dataset = load_data(root=self.configs.root_path, data_name=self.configs.dataset)
        data = dataset[0]
        data.tokens = get_eigen_tokens(data, self.configs.embed_dim, self.device)
        train_loader = ExtractNodeLoader(data, input_nodes=data.train_mask, batch_size=self.configs.batch_size,
                                   num_neighbors=self.configs.num_neighbors,
                                   capacity=self.configs.capacity)
        val_loader = ExtractNodeLoader(data, input_nodes=data.val_mask, batch_size=self.configs.batch_size,
                                         num_neighbors=self.configs.num_neighbors,
                                         capacity=self.configs.capacity)
        test_loader = ExtractNodeLoader(data, input_nodes=data.test_mask, batch_size=self.configs.batch_size,
                                         num_neighbors=self.configs.num_neighbors,
                                         capacity=self.configs.capacity)
        if split == 'test':
            return test_loader
        return dataset, train_loader, val_loader, test_loader

    def train(self):
        dataset, train_loader, val_loader, test_loader = self.load_data("train")
        total_test_acc = []
        total_test_weighted_f1 = []
        total_test_macro_f1 = []
        for t in range(self.configs.exp_iters):
            self.nc_model = self.load_model()
            self.nc_model.train()
            optimizer = Adam(self.nc_model.parameters(), lr=self.configs.lr_nc,
                                       weight_decay=self.configs.weight_decay_nc)
            early_stop = EarlyStopping(self.configs.patience)
            for epoch in range(self.configs.nc_epochs):
                epoch_loss = []
                trues = []
                preds = []

                for data in tqdm(train_loader):
                    data = data.to(self.device)
                    loss, pred, true = self.train_step(data, optimizer)
                    epoch_loss.append(loss)
                    trues.append(true)
                    preds.append(pred)
                trues = np.concatenate(trues, axis=-1)
                preds = np.concatenate(preds, axis=-1)
                train_loss = np.mean(epoch_loss)
                train_acc = cal_accuracy(preds, trues)

                self.logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc * 100: .2f}%")

                if epoch % self.configs.val_every == 0:
                    val_loss, val_acc, val_weighted_f1, val_macro_f1 = self.val(val_loader)
                    self.logger.info(f"Epoch {epoch}: val_loss={val_loss}, "
                                     f"val_acc={val_acc * 100: .2f}%,"
                                     f"val_weighted_f1={val_weighted_f1 * 100: .2f},"
                                     f"val_macro_f1={val_macro_f1 * 100: .2f}%")
                    early_stop(val_loss, self.nc_model, self.configs.checkpoints, self.configs.task_model_path)
                    if early_stop.early_stop:
                        print("---------Early stopping--------")
                        break
            test_acc, weighted_f1, macro_f1 = self.test(test_loader)
            self.logger.info(f"test_acc={test_acc * 100: .2f}%, "
                             f"weighted_f1={weighted_f1 * 100: .2f},"
                             f"macro_f1={macro_f1 * 100: .2f}%")
            total_test_acc.append(test_acc)
            total_test_weighted_f1.append(weighted_f1)
            total_test_macro_f1.append(macro_f1)
        mean, std = np.mean(total_test_acc), np.std(total_test_acc)
        self.logger.info(f"Evaluation Acc is {mean * 100: .2f}% +- {std * 100: .2f}%")
        mean, std = np.mean(total_test_weighted_f1), np.std(total_test_weighted_f1)
        self.logger.info(f"Evaluation weighted F1 is {mean * 100: .2f}% +- {std * 100: .2f}%")
        mean, std = np.mean(total_test_macro_f1), np.std(total_test_macro_f1)
        self.logger.info(f"Evaluation macro F1 is {mean * 100: .2f}% +- {std * 100: .2f}%")


    def train_step(self, data, optimizer):
        optimizer.zero_grad()
        out = self.nc_model(data)
        loss, preds, trues = self.cal_loss(out, data.y, data.batch_size)
        loss.backward()
        optimizer.step()
        return loss.item(), preds, trues

    def val(self, val_loader):
        self.nc_model.eval()
        val_loss = []
        trues = []
        preds = []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                out = self.nc_model(data)
                loss, pred, true = self.cal_loss(out, data.y, data.batch_size)
                val_loss.append(loss.item())
                trues.append(true)
                preds.append(pred)
        trues = np.concatenate(trues, axis=-1)
        preds = np.concatenate(preds, axis=-1)
        acc = cal_accuracy(preds, trues)
        weighted_f1, macro_f1 = cal_F1(preds, trues)
        self.nc_model.train()
        return np.mean(val_loss), acc, weighted_f1, macro_f1

    def test(self, test_loader=None):
        test_loader = self.load_data("test") if test_loader is None else test_loader
        self.nc_model.eval()
        self.logger.info("--------------Testing--------------------")
        path = os.path.join(self.configs.checkpoints, self.configs.task_model_path)
        self.logger.info(f"--------------Loading from {path}--------------------")
        self.nc_model.load_state_dict(torch.load(path))
        trues = []
        preds = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out = self.nc_model(data)
                loss, pred, true = self.cal_loss(out, data.y, data.batch_size)
                trues.append(true)
                preds.append(pred)
        trues = np.concatenate(trues, axis=-1)
        preds = np.concatenate(preds, axis=-1)
        test_acc = cal_accuracy(preds, trues)
        weighted_f1, macro_f1 = cal_F1(preds, trues)
        self.logger.info(f"test_acc={test_acc * 100: .2f}%, "
                         f"weighted_f1={weighted_f1 * 100: .2f},"
                         f"macro_f1={macro_f1 * 100: .2f}%")
        return test_acc, weighted_f1, macro_f1

    def cal_loss(self, output, label, batch_size):
        out = output[:batch_size]
        y = label[:batch_size].reshape(-1)
        loss = F.cross_entropy(out, y)
        pred = out.argmax(dim=-1).detach().cpu().numpy()
        return loss, pred, y.detach().cpu().numpy()


class LinkPrediction(SupervisedExp):
    def __init__(self, configs, pretrained_model=None, load=False, finetune=False):
        super(LinkPrediction, self).__init__(configs, pretrained_model, load, finetune)
        self.lp_model = self.load_model()

    def load_data(self, split):
        dataset = load_data(root=self.configs.root_path, data_name=self.configs.dataset)
        train_data, val_data, test_data = RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=False,
                                                          add_negative_train_samples=False)(dataset[0])
        train_data.tokens = get_eigen_tokens(train_data, self.configs.embed_dim, self.device)
        val_data.tokens = train_data.tokens
        test_data.tokens = train_data.tokens
        train_loader = ExtractLinkLoader(train_data, batch_size=self.configs.batch_size,
                                   num_neighbors=self.configs.num_neighbors,
                                         neg_sampling_ratio=1.,
                                   capacity=self.configs.capacity)
        val_loader = ExtractLinkLoader(val_data, batch_size=self.configs.batch_size,
                                     num_neighbors=self.configs.num_neighbors,
                                       neg_sampling_ratio=1.,
                                     capacity=self.configs.capacity)
        test_loader = ExtractLinkLoader(test_data, batch_size=self.configs.batch_size,
                                     num_neighbors=self.configs.num_neighbors,
                                        neg_sampling_ratio=1.,
                                     capacity=self.configs.capacity)
        if split == 'test':
            return test_loader
        return train_data, train_loader, val_loader, test_loader

    def load_model(self):
        lp_model = LinkPredHead(self.pretrained_model,
                                2 * self.configs.embed_dim + input_dim_dict[self.configs.dataset],
                                self.configs.embed_dim_lp).to(self.device)
        return lp_model

    def train(self):
        train_data, train_loader, val_loader, test_loader = self.load_data(None)
        total_test_auc, total_test_ap = [], []
        for _ in range(self.configs.exp_iters):
            self.lp_model = self.load_model()
            self.lp_model.train()
            optimizer = Adam(self.lp_model.parameters(), lr=self.configs.lr_lp,
                             weight_decay=self.configs.weight_decay_lp)
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
            total_test_auc.append(test_auc)
            total_test_ap.append(test_ap)
        mean_auc, std_auc = np.mean(total_test_auc), np.std(total_test_auc)
        mean_ap, std_ap = np.mean(total_test_ap), np.std(total_test_ap)
        self.logger.info(f"Evaluation AUC={mean_auc * 100: .2f}% +- {std_auc * 100: .2f}%, "
                         f"Evaluation AP={mean_ap * 100: .2f}% +- {std_ap * 100: .2f}%")

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