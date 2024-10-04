import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import negative_sampling, dropout_edge


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
    def __init__(self, pretrained_model, in_dim, num_cls):
        """

        :param in_dim: input dimension of three components
        :param num_cls: number of classes
        """
        super(NodeClsHead, self).__init__()
        self.pretrained_model = pretrained_model
        self.head = GCN(2, in_dim, 32, num_cls, drop_edge=0.5, drop_feats=0.2)
    
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


class GraphClsHead(nn.Module):
    def __init__(self, pretrained_model, in_dim, num_cls):
        super(GraphClsHead, self).__init__()
        self.pretrained_model = pretrained_model
        self.head = nn.Linear(in_dim, num_cls)

    def forward(self, data):
        x_E, x_H, x_S = self.pretrained_model(data)
        manifold_H = self.pretrained_model.manifold_H
        manifold_S = self.pretrained_model.manifold_S
        m_E = x_E.mean(0)
        m_H = manifold_H.Frechet_mean(x_H, dim=0)
        m_H = manifold_H.logmap0(m_H)
        m_S = manifold_S.Frechet_mean(x_S, dim=0)
        m_S = manifold_S.logmap0(m_S)
        x = torch.concat([m_E, m_H, m_S], dim=-1).mean(0)
        return self.head(x)


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
        x = torch.concat([x_E, x_h, x_s], dim=-1)
        x = self.head(x)

        neg_edge_index = negative_sampling(
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            data.edge_label,
            data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        x_src = x[edge_label_index[0]]
        x_dst = x[edge_label_index[1]]
        score = F.cosine_similarity(x_src, x_dst, dim=-1)

        return score, edge_label


class ShotNCHead(nn.Module):
    def __init__(self, pretrained_model, cls_embeddings, in_dim, cls_dim):
        """

        :param in_dim: input dimension of three components
        :param num_cls: number of classes
        """
        super(ShotNCHead, self).__init__()
        self.pretrained_model = pretrained_model
        self.head = GCNConv(in_dim, cls_dim)
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
        x = torch.concat([x_E, x_h, x_s], dim=-1)
        x = self.head(x, data.edge_index)
        out = F.cosine_similarity(x.unsqueeze(1), self.cls_embeddings.unsqueeze(0), dim=-1)
        return out