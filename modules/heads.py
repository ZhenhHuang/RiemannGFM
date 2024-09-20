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