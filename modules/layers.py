import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_sum
import math
from manifolds import Lorentz, Sphere


class EuclideanEncoder(nn.Module):
    """
    This module can be implemented by any unstructured model, e.g. MLPs, Transformers, LLMs.
    Here we use MLP as a basic example and fine-tune it.
    But for other models like Transformer and LLM, we can pretrain it.
    """
    def __init__(self, in_dim, out_dim, bias=True, activation=F.relu, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.proj = nn.Linear(out_dim, out_dim, bias=bias)
        self.drop = dropout

    def forward(self, x):
        x = self.activation(self.lin(x))
        x = self.proj(F.dropout(x, p=self.drop, training=self.training))
        return x


class ManifoldEncoder(nn.Module):
    def __init__(self, manifold, in_dim, out_dim, bias=True, activation=None, dropout=0.1):
        super().__init__()
        self.manifold = manifold
        self.lin = ConstCurveLinear(manifold, in_dim + 1, out_dim + 1, bias=bias, dropout=dropout, activation=activation)
        self.proj = ConstCurveLinear(manifold, out_dim + 1, out_dim + 1, bias=bias, dropout=dropout, activation=activation)

    def forward(self, x):
        o = torch.zeros_like(x).to(x.device)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.expmap0(x)
        x = self.lin(x)
        x = self.proj(x)
        return x


class ConstCurveLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.0,
                 scale=10,
                 fixscale=False,
                 activation=None):
        super().__init__()
        self.manifold = manifold
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)
        self.sign = -1. if isinstance(manifold, Lorentz) else 1.

    def forward(self, x):
        if self.activation is not None:
            x = self.activation(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1 if isinstance(self.manifold, Lorentz)  \
        else x.narrow(-1, 0, 1)
        scale = self.sign * (1. / self.manifold.k - time * time) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class ConstCurveAgg(nn.Module):
    """
    Lorentz aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout=0.0, use_att=False):
        super(ConstCurveAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.key_linear = ConstCurveLinear(manifold, in_features, in_features)
            self.query_linear = ConstCurveLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))
        self.neg_dist = lambda x, y: 2 + 2 * manifold.cinner(x, y) if isinstance(manifold, Lorentz) \
            else lambda x, y: -manifold.dist(x, y) ** 2
        self.sign = -1. if isinstance(manifold, Lorentz) else 1.

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            att_adj = 2 + 2 * self.manifold.cinner(query[dst], key[src])
            att_adj = att_adj / self.scale + self.bias
            att_adj = torch.sigmoid(att_adj)
            support_t = scatter_sum(att_adj * x[src], dst, dim=0)
        else:
            support_t = scatter_sum(x[src], dst, dim=0)

        denorm = self.sign * self.manifold.inner(None, support_t, keepdim=True)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        output = 1. / self.manifold.k.sqrt() * support_t / denorm
        return output


# if __name__ == '__main__':
    # manifold = Lorentz()
    # manifold = Sphere()
    # x = manifold.random_normal((3, 4))
    # edge_index = torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]], dtype=torch.long)
    # lin = ConstCurveLinear(manifold, 4, 4)
    # y = lin(x)
    # print(manifold.check_point_on_manifold(y))
    # agg = ConstCurveAgg(manifold, 4, 0.1, True)
    # z = agg(y, edge_index)
    # print(z.shape)
    # print(manifold.check_point_on_manifold(z))

    # x = torch.randn((3, 4))
    # encoder = ManifoldEncoder(manifold, 4, 4)
    # y = encoder(x)
    # print(manifold.check_point_on_manifold(y))