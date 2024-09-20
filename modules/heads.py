import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class NodeClsHead(nn.modules):
    def __init__(self, in_dim, num_cls):
        """

        :param in_dim: input dimension of three components
        :param num_cls: number of classes
        """
        super(NodeClsHead, self).__init__()
        self.head = nn.Linear(in_dim, num_cls)

    def forward(self, x_tuple):
        x = torch.concat(x_tuple, dim=-1)
        return self.head(x)


class GraphClsHead(nn.modules):
    def __init__(self, in_dim, num_cls):
        super(GraphClsHead, self).__init__()
        self.head = nn.Linear(in_dim, num_cls)

    def forward(self, x_tuple):
        x = torch.concat(x_tuple, dim=-1).mean(0)
        return self.head(x)


class LinkPredHead(nn.modules):
    def __init__(self, in_dim, out_dim, r, s):
        super(LinkPredHead, self).__init__()
        self.head = nn.Linear(in_dim, out_dim)
        self.r = r
        self.s = s

    def forward(self, x_tuple, pos_edge_index, neg_edge_index):
        x = self.head(torch.cat(x_tuple, dim=-1))
        x_src = x[pos_edge_index[0]]
        x_dst = x[pos_edge_index[1]]
        pos_score = F.cosine_similarity(x_src, x_dst, dim=-1)

        x_src = x[neg_edge_index[0]]
        x_dst = x[neg_edge_index[1]]
        neg_score = F.cosine_similarity(x_src, x_dst, dim=-1)

        return pos_score, neg_score
