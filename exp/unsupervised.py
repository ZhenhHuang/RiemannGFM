import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn import metrics
from torch.optim import Adam
import numpy as np
from modules import *
from utils.train_utils import EarlyStopping, act_fn
from utils.logger import create_logger
from data import *
import os
from tqdm import tqdm


class NodeCluster:
    def __init__(self, configs, load=True):
        self.configs = configs
        self.logger = create_logger(self.configs.log_dir + self.configs.log_name)
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        model = GeoGFM(n_layers=self.configs.n_layers, in_dim=self.configs.embed_dim,
                       hidden_dim=self.configs.hidden_dim, embed_dim=self.configs.embed_dim,
                       bias=self.configs.bias,
                       dropout=self.configs.dropout, activation=act_fn(self.configs.activation))
        if load:
            path = os.path.join(self.configs.checkpoints, self.configs.pretrained_model_path) + ".pt"
            self.logger.info(f"---------------Loading pretrained models from {path}-------------")

            pretrained_dict = torch.load(path)
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        self.model = model.to(self.device)

    def load_data(self):
        dataset = load_data(root=self.configs.root_path, data_name=self.configs.dataset)
        data = dataset[0]
        data.tokens = get_eigen_tokens(data, self.configs.embed_dim, self.device)
        loader = ExtractNodeLoader(data, batch_size=self.configs.batch_size,
                                   num_neighbors=self.configs.num_neighbors,
                                   capacity=self.configs.capacity)
        return dataset, loader

    def run(self, train=False):
        if train:
            dataset, loader = self.train()
        else:
            dataset, loader = self.load_data()
        self.test(dataset, loader)

    def train(self):
        dataset, loader = self.load_data()
        optimizer = Adam(self.model.parameters(), lr=self.configs.lr_clu, weight_decay=self.configs.weight_decay_clu)
        early_stop = EarlyStopping(self.configs.patience)
        for epoch in range(self.configs.clu_epochs):
            epoch_loss = []
            for data in tqdm(loader):
                optimizer.zero_grad()
                data = data.to(self.device)
                output = self.model(data)
                loss = self.model.loss(output)
                if torch.isnan(loss).item():
                    continue
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            train_loss = np.mean(epoch_loss)
            self.logger.info(f"Epoch {epoch}: train_loss={train_loss}")

            if epoch % self.configs.val_every == 0:
                nmi, ari = self.test(dataset, loader)
                self.logger.info(f"NMI={nmi * 100: .2f}%, ARI={ari * 100: .2f}")

            early_stop(train_loss, self.model, self.configs.checkpoints, self.configs.task_model_path)
            if early_stop.early_stop:
                print("---------Early stopping--------")
                break
        return dataset, loader

    def test(self, dataset, loader):
        embeddings = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                x_E, x_H, x_S = self.model(data)
                # x_E = x_E[: data.batch_size]
                # x_H = x_H[: data.batch_size]
                x_S = x_S[: data.batch_size]
                # manifold_H = self.model.manifold_H
                manifold_S = self.model.manifold_S
                # x_h = manifold_H.logmap0(x_H)
                x_s = manifold_S.logmap0(x_S)
                # x = torch.concat([x_E, x_h, x_s], dim=-1)
                embeddings.append(x_s)
                trues.append(data.y[: data.batch_size].reshape(-1))
        embeddings = torch.concat(embeddings, dim=0).cpu().numpy()
        trues = torch.concat(trues, dim=0).cpu().numpy()
        kmeans = KMeans(n_clusters=class_num_dict[self.configs.dataset])
        preds = kmeans.fit_predict(embeddings)
        nmi = metrics.normalized_mutual_info_score(trues, preds)
        ari = metrics.adjusted_rand_score(trues, preds)
        self.logger.info(f"NMI={nmi * 100: .2f}%, ARI={ari * 100: .2f}")
        return nmi, ari
