import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv


class NodeClsHead(nn.Module):
    def __init__(self, pretrained_model, in_dim, num_cls):
        """

        :param in_dim: input dimension of three components
        :param num_cls: number of classes
        """
        super(NodeClsHead, self).__init__()
        self.pretrained_model = pretrained_model
        self.head = GCNConv(in_dim, num_cls, bias=False)
    
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

        pos_edge_index = data.edge_index
        neg_edge_index = data.neg_edge_index
        x_src = x[pos_edge_index[0]]
        x_dst = x[pos_edge_index[1]]
        pos_score = F.cosine_similarity(x_src, x_dst, dim=-1)

        x_src = x[neg_edge_index[0]]
        x_dst = x[neg_edge_index[1]]
        neg_score = F.cosine_similarity(x_src, x_dst, dim=-1)

        return pos_score, neg_score
