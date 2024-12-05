import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, dropout_edge
from torch_scatter import scatter_mean


class GCN(nn.Module):
    def __init__(self, n_layers, in_features, hidden_features, out_features, drop_edge=0.5, drop_feats=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_features, hidden_features))
        for _ in range(n_layers - 2):
            self.layers.append(GCNConv(hidden_features, hidden_features))
        self.layers.append(GCNConv(hidden_features, out_features))
        self.drop_edge = drop_edge
        self.drop = nn.Dropout(drop_feats)

    def forward(self, x, edge_index):
        edge = dropout_edge(edge_index, self.drop_edge, training=self.training)[0]
        for layer in self.layers[:-1]:
            x = self.drop(F.relu(layer(x, edge)))
        x = self.layers[-1](x, edge)
        return x


class NodeClsHead(nn.Module):
    def __init__(self, pretrained_model, in_dim, hidden_dim, num_cls, drop_edge, drop_feats):
        """

        :param in_dim: input dimension of three components
        :param num_cls: number of classes
        """
        super(NodeClsHead, self).__init__()
        self.pretrained_model = pretrained_model
        self.head = GCN(2, in_dim, hidden_dim, num_cls, drop_edge=drop_edge, drop_feats=drop_feats)

    def forward(self, data):
        """

        :param data
        :return:
        """
        x_E, x_H, x_S = self.pretrained_model(data)
        manifold_H = self.pretrained_model.manifold_H
        manifold_S = self.pretrained_model.manifold_S
        x_h = manifold_H.logmap0(x_H)
        x_s = manifold_S.logmap0(x_S)
        x = torch.concat([data.x, x_h, x_s], dim=-1)
        return self.head(x, data.edge_index)


class LinkPredHead(nn.Module):
    def __init__(self, pretrained_model, in_dim, out_dim):
        super(LinkPredHead, self).__init__()
        self.pretrained_model = pretrained_model
        self.head = nn.Linear(in_dim, out_dim)

    def forward(self, data):
        x_E, x_H, x_S = self.pretrained_model(data)
        manifold_H = self.pretrained_model.manifold_H
        manifold_S = self.pretrained_model.manifold_S
        x_h = manifold_H.logmap0(x_H)
        x_s = manifold_S.logmap0(x_S)
        x = torch.concat([data.x, x_h, x_s], dim=-1)
        x = self.head(x)

        x_src = x[data.edge_label_index[0]]
        x_dst = x[data.edge_label_index[1]]
        score = F.cosine_similarity(x_src, x_dst, dim=-1)

        return score, data.edge_label


class ShotNCHead(nn.Module):
    def __init__(self, pretrained_model, cls_embeddings, in_dim, hidden_dim, cls_dim, drop_edge, drop_feats):
        """

        :param in_dim: input dimension of three components
        :param num_cls: number of classes
        """
        super(ShotNCHead, self).__init__()
        self.pretrained_model = pretrained_model
        self.head = GCN(2, in_dim, hidden_dim, cls_dim, drop_edge=drop_edge, drop_feats=drop_feats)
        self.cls_embeddings = cls_embeddings

    def forward(self, data):
        """

        :param data
        :return:
        """
        x_E, x_H, x_S = self.pretrained_model(data)
        manifold_H = self.pretrained_model.manifold_H
        manifold_S = self.pretrained_model.manifold_S
        x_h = manifold_H.logmap0(x_H)
        x_s = manifold_S.logmap0(x_S)
        x = torch.concat([data.x, x_h, x_s], dim=-1)
        x = self.head(x, data.edge_index)
        out = F.cosine_similarity(x.unsqueeze(1), self.cls_embeddings.unsqueeze(0), dim=-1)
        return out