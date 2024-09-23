import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from modules.layers import EuclideanEncoder, ManifoldEncoder
from modules.basics import HyperbolicStructureLearner, SphericalStructureLearner
from manifolds import Lorentz, Sphere


class GeoGFM(nn.Module):
    def __init__(self, n_layers, in_dim, out_dim, bias, activation, dropout):
        super(GeoGFM, self).__init__()
        self.manifold_H = Lorentz()
        self.manifold_S = Sphere()
        self.init_block = InitBlock(self.manifold_H, self.manifold_S, in_dim, out_dim, bias, activation, dropout)
        self.blocks = nn.ModuleList([])
        for i in range(n_layers):
            self.blocks.append(StructuralBlock(self.manifold_H, self.manifold_S, out_dim))

    def forward(self, data):
        """

        :param data: Dataset for a graph contains batched sub-graphs and sub-trees.
        :return: z: node product representations
        """
        x = data.x.clone()
        x_E, x_H, x_S = self.init_block(x)
        for i, block in enumerate(self.blocks):
            x_H, x_S = block((x_H, x_S), data)
        return x_E, x_H, x_S


class InitBlock(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, out_dim, bias, activation, dropout):
        super(InitBlock, self).__init__()
        self.Euc_init = EuclideanEncoder(in_dim, out_dim, bias, activation, dropout)
        self.Hyp_init = ManifoldEncoder(manifold_H, in_dim, out_dim, bias, None, 0.)
        self.Sph_init = ManifoldEncoder(manifold_S, in_dim, out_dim, bias, None, 0.)

    def forward(self, x):
        """

        :param x: raw features
        :return: (E, H, S) Manifold initial representations
        """
        E = self.Euc_init(x)
        H = self.Hyp_init(x)
        S = self.Sph_init(x)
        return E, H, S


class StructuralBlock(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim):
        super(StructuralBlock, self).__init__()
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        self.Hyp_learner = HyperbolicStructureLearner(self.manifold_H, in_dim=in_dim)
        self.Sph_learner = SphericalStructureLearner(self.manifold_S, in_dim=in_dim)

    def forward(self, x_tuple, data):
        """

        :param x_tuple: (x_H, x_S)
        :param data: Dataset for a graph contains batched sub-graphs and sub-trees
        :return: x_tuple: (x_H, x_S)
        """
        x_H, x_S = x_tuple
        x_H = self.Hyp_learner(x_H, data.batch_tree[0])
        x_S = self.Sph_learner(x_S, data.batch_data[0])
        return x_H, x_S


# if __name__ == '__main__':
#     from data.graph_exacters import graph_exacter
#     from torch_geometric.datasets import KarateClub
#     dataset = KarateClub()
#     data = dataset.get(0)
#     data_dict = {}
#     for i in range(2):
#         data_dict[i + 1] = graph_exacter(data, k_hop=i + 1)
#     model = GeoGFM(2, 34, 5, True, F.relu, 0.)
#     y = model(data.x, data_dict)