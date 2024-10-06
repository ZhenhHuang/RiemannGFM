import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from modules import *
import gensim.downloader as api
from utils.train_utils import EarlyStopping, act_fn
from utils.evall_utils import cal_accuracy, cal_F1
from utils.logger import create_logger
from data import load_data, input_dim_dict, ExtractNodeLoader, get_eigen_tokens
from data.mappings import class_maps
import os
from tqdm import tqdm
import re


class FewShotNC:
    def __init__(self, configs, load: bool = False):
        self.configs = configs
        self.load = load
        self.logger = create_logger(self.configs.log_dir + self.configs.log_name)
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.load_word2vec()
        self.supp_sets = self.configs.pretrain_dataset
        self.query_set = self.configs.query_set
        supp_embed_dict, query_embed_dict = self.get_class_embedding(self.supp_sets, self.query_set)
        self.merge_embeddings(supp_embed_dict, query_embed_dict)
        self.load_model()
        self.k_shot = self.configs.k_shot

    def load_model(self):
        pretrained_model = GeoGFM(n_layers=self.configs.n_layers, in_dim=self.configs.embed_dim,
                                  hidden_dim=self.configs.hidden_dim, embed_dim=self.configs.embed_dim,
                                  bias=self.configs.bias,
                                  dropout=self.configs.dropout, activation=act_fn(self.configs.activation))
        if self.load:
            path = os.path.join(self.configs.checkpoints, self.configs.pretrained_model_path) + ".pt"
            self.logger.info(f"---------------Loading pretrained models from {path}-------------")
            pretrained_dict = torch.load(path)
            model_dict = pretrained_model.state_dict()
            model_dict.update(pretrained_dict)
            pretrained_model.load_state_dict(model_dict)

            self.logger.info("----------Freezing weights-----------")
            for module in pretrained_model.modules():
                for param in module.parameters():
                    param.requires_grad = False
            pretrained_model = pretrained_model.to(self.device)
        self.nc_model = ShotNCHead(pretrained_model, self.class_embeddings,
                                   2 * self.configs.embed_dim + input_dim_dict[self.query_set],
                               100).to(self.device)

    def load_data(self, data_name, finetune=False):
        """

        :param data_name: name of dataset
        :param finetune: use for k-shot fine-tuning
        :return: Data and Dataloader
        """
        if self.k_shot == 0:
            num_per_class = None
        else:
            num_per_class = self.k_shot
        if data_name == self.query_set:
            if finetune:
                dataset = load_data(root=self.configs.root_path, data_name=self.query_set, num_per_class=num_per_class)
                data = self.convert_label(dataset[0].clone(), self.query_set)
                data.tokens = get_eigen_tokens(data, self.configs.embed_dim, self.device)
                query_loader = ExtractNodeLoader(data, input_nodes=data.train_mask, batch_size=self.configs.batch_size,
                                                 num_neighbors=self.configs.num_neighbors,
                                                 capacity=self.configs.capacity)
            else:
                dataset = load_data(root=self.configs.root_path, data_name=self.query_set)
                data = self.convert_label(dataset[0].clone(), self.query_set)
                data.tokens = get_eigen_tokens(data, self.configs.embed_dim, self.device)
                query_loader = ExtractNodeLoader(data, input_nodes=data.test_mask, batch_size=self.configs.batch_size,
                                                num_neighbors=self.configs.num_neighbors,
                                                capacity=self.configs.capacity)
            return data, query_loader

        dataset = load_data(root=self.configs.root_path, data_name=data_name)
        data = self.convert_label(dataset[0].clone(), data_name)
        data.tokens = get_eigen_tokens(data, self.configs.embed_dim, self.device)
        dataloader = ExtractNodeLoader(data, batch_size=self.configs.batch_size,
                                       num_neighbors=self.configs.num_neighbors,
                                       capacity=self.configs.capacity)
        return data, dataloader

    def train(self, load_trained_model=False):
        total_test_acc = []
        total_test_weighted_f1 = []
        total_test_macro_f1 = []
        for t in range(self.configs.exp_iters):
            self.load_model()
            if self.k_shot > 0:
                self.logger.info(f"----------Fine-tuning on {self.query_set}---------")
                self._train_step(load=load_trained_model, data_name=self.query_set,
                                train_epochs=self.configs.shot_epochs, finetune=True,
                                lr=self.configs.lr_few_nc)
            test_acc, weighted_f1, macro_f1 = self.test()
            total_test_acc.append(test_acc)
            total_test_weighted_f1.append(weighted_f1)
            total_test_macro_f1.append(macro_f1)
        mean, std = np.mean(total_test_acc), np.std(total_test_acc)
        self.logger.info(f"Evaluation Acc is {mean * 100: .2f}% +- {std * 100: .2f}%")
        mean, std = np.mean(total_test_weighted_f1), np.std(total_test_weighted_f1)
        self.logger.info(f"Evaluation weighted F1 is {mean * 100: .2f}% +- {std * 100: .2f}%")
        mean, std = np.mean(total_test_macro_f1), np.std(total_test_macro_f1)
        self.logger.info(f"Evaluation macro F1 is {mean * 100: .2f}% +- {std * 100: .2f}%")

    def test(self):
        data, test_loader = self.load_data(self.configs.query_set)
        self.nc_model.eval()
        self.logger.info("--------------Testing--------------------")
        path = os.path.join(self.configs.checkpoints, self.configs.trained_model_path_FSL) + ".pt"
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

    def _train_step(self, load, data_name, train_epochs, lr, finetune=False):
        if load:
            load_path = os.path.join(self.configs.checkpoints, self.configs.trained_model_path_FSL) + f".pt"
            self.logger.info(f"---------------Loading trained models from {load_path}-------------")
            pretrained_dict = torch.load(load_path)
            model_dict = self.nc_model.state_dict()
            model_dict.update(pretrained_dict)
            self.nc_model.load_state_dict(model_dict)

        path = os.path.join(self.configs.checkpoints, self.configs.trained_model_path_FSL)
        data, dataloader = self.load_data(data_name, finetune=finetune)

        optimizer = Adam(self.nc_model.parameters(), lr=lr, weight_decay=self.configs.weight_decay)
        for epoch in range(train_epochs):
            epoch_loss = []
            trues = []
            preds = []
            for data in tqdm(dataloader):
                optimizer.zero_grad()
                data = data.to(self.device)
                out = self.nc_model(data)
                loss, pred, true = self.cal_loss(out, data.y, data.batch_size)
                if torch.isnan(loss).item():
                    continue
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
                trues.append(true)
                preds.append(pred)
            trues = np.concatenate(trues, axis=-1)
            preds = np.concatenate(preds, axis=-1)
            train_loss = np.mean(epoch_loss)
            train_acc = cal_accuracy(preds, trues)
            self.logger.info(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc * 100: .2f}%")
            # self.logger.info(f"---------------Saving pretrained models to {path}_{epoch}.pt-------------")
            # torch.save(self.nc_model.state_dict(), path + f"_{epoch}.pt")  # save epoch every
            self.logger.info(f"---------------Saving pretrained models to {path}.pt-------------")
            torch.save(self.nc_model.state_dict(), path + f".pt")  # save for next training

    def load_word2vec(self, word2vec_path='glove-wiki-gigaword-100'):
        self.word2vec = api.load(word2vec_path)

    def get_class_embedding(self, supp_sets: list, query_set: str):
        sp_pattern = r'[ -.%]+'
        supp_embed_dict = {}
        for data_name in supp_sets:
            supp_embed_dict[data_name] = {}
            cls_map = class_maps[data_name]
            for k, word in cls_map.items():
                text = re.split(sp_pattern, word)
                supp_embed_dict[data_name][k] = np.mean([self.word2vec[t.lower()] for t in text], axis=0)

        query_embed_dict = {}
        cls_map = class_maps[query_set]
        for k, word in cls_map.items():
            text = re.split(sp_pattern, word)
            query_embed_dict[k] = np.mean([self.word2vec[t.lower()] for t in text], axis=0)

        return supp_embed_dict, query_embed_dict

    def merge_embeddings(self, supp_embed_dict, query_embed_dict):
        merge_embed_dict = {}
        offset_dict = {}
        for i, (data_name, d) in enumerate(supp_embed_dict.items()):
            offset = len(merge_embed_dict)
            offset_dict[data_name] = offset
            for key, value in d.items():
                merge_embed_dict[key + offset] = value
        offset = len(merge_embed_dict)
        offset_dict[self.query_set] = offset
        for key, value in query_embed_dict.items():
            merge_embed_dict[key + offset] = value
        merge_embed = np.stack([em for em in merge_embed_dict.values()], axis=0)
        self.class_embeddings = torch.tensor(merge_embed).to(self.device)
        self.offset_dict = offset_dict

    def convert_label(self, data, data_name):
        data.y = data.y + self.offset_dict[data_name]
        return data

    def cal_loss(self, output, label, batch_size):
        out = output[:batch_size]
        y = label[:batch_size].reshape(-1)
        loss = F.cross_entropy(out, y)
        pred = out.argmax(dim=-1).detach().cpu().numpy()
        return loss, pred, y.detach().cpu().numpy()