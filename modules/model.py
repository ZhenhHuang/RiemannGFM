import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from layers import EuclideanEncoder, ManifoldEncoder
from basics import HyperbolicStructureLearner, SphericalStructureLearner
from manifolds import Lorentz, Sphere


class GeoGFM(nn.Module):
    def __init__(self, in_dim, out_dim, bias, activation, dropout):
        super(GeoGFM, self).__init__()
        self.manifold_H = Lorentz()
        self.manifold_S = Sphere()
        self.init_block = InitBlock(self.manifold_H, self.manifold_S, in_dim, out_dim, bias, activation, dropout)



    def forward(self, x):
        x_E, x_H, x_S = self.init_block(x)


class InitBlock(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, out_dim, bias, activation, dropout):
        super(InitBlock, self).__init__()
        self.Euc_init = EuclideanEncoder(in_dim, out_dim, bias, activation, dropout)
        self.Hyp_init = ManifoldEncoder(manifold_H, in_dim, out_dim, bias, activation, dropout)
        self.Sph_init = ManifoldEncoder(manifold_S, in_dim, out_dim, bias, activation, dropout)

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
    def __init__(self):
        super(StructuralBlock, self).__init__()

    def forward(self, x_tuple):
        pass