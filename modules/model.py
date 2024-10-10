import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from modules.layers import EuclideanEncoder, ManifoldEncoder
from modules.basics import HyperbolicStructureLearner, SphericalStructureLearner
from manifolds import Lorentz, Sphere, ProductSpace


class GeoGFM(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim, embed_dim, bias, activation, dropout):
        super(GeoGFM, self).__init__()
        self.manifold_H = Lorentz()
        self.manifold_S = Sphere()
        self.product = ProductSpace(*[(self.manifold_H, embed_dim),
                                      (self.manifold_S, embed_dim)])
        self.init_block = InitBlock(self.manifold_H, self.manifold_S,
                                    in_dim, hidden_dim, embed_dim, bias,
                                    activation, dropout)
        self.blocks = nn.ModuleList([])
        for i in range(n_layers):
            self.blocks.append(StructuralBlock(self.manifold_H, self.manifold_S,
                                               embed_dim, hidden_dim, embed_dim, dropout))
        self.proj = nn.Sequential(nn.Linear(2 * embed_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, embed_dim))

    def forward(self, data):
        """

        :param data: Dataset for a graph contains batched sub-trees.
        :return: z: node product representations
        """
        x = data.x.clone()
        x_E, x_H, x_S = self.init_block(x, data.edge_index, data.tokens(data.n_id))  # [N, Hidden]
        for i, block in enumerate(self.blocks):
            x_E, x_H, x_S = block((x_E, x_H, x_S), data)
        return x_E, x_H, x_S

    def loss(self, x_tuple):
        """

        :param x_tuple: (x_E, x_H, x_S)
        :return:
        """

        x_E, x_H, x_S = x_tuple

        H_E = self.manifold_H.proju(x_H, x_E)
        S_E = self.manifold_H.proju(x_S, x_E)

        H_E = self.manifold_H.transp0back(x_H, H_E)
        S_E = self.manifold_S.transp0back(x_S, S_E)

        log0_H = self.manifold_H.logmap0(x_H)
        log0_S = self.manifold_S.logmap0(x_S)
        H_E = self.proj(torch.cat([log0_H, H_E], dim=-1))
        S_E = self.proj(torch.cat([log0_S, S_E], dim=-1))
        loss = self.cal_cl_loss(H_E, S_E)

        return loss

    def cal_cl_loss(self, x1, x2):
        EPS = 1e-6
        norm1 = x1.norm(dim=-1)
        norm2 = x2.norm(dim=-1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / (torch.einsum('i,j->ij', norm1, norm2) + EPS)
        sim_matrix = torch.exp(sim_matrix / 0.2)
        pos_sim = sim_matrix.diag()
        loss_1 = pos_sim / (sim_matrix.sum(dim=-2) + EPS)
        loss_2 = pos_sim / (sim_matrix.sum(dim=-1) + EPS)

        loss_1 = -torch.log(loss_1).mean()
        loss_2 = -torch.log(loss_2).mean()
        loss = (loss_1 + loss_2) / 2.
        return loss


class InitBlock(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, bias, activation, dropout):
        super(InitBlock, self).__init__()
        self.Euc_init = EuclideanEncoder(in_dim, hidden_dim, out_dim, bias, activation, dropout)
        self.Hyp_init = ManifoldEncoder(manifold_H, in_dim, hidden_dim, out_dim, bias, None, 0.)
        self.Sph_init = ManifoldEncoder(manifold_S, in_dim, hidden_dim, out_dim, bias, None, 0.)

    def forward(self, x, edge_index, tokens):
        """

        :param tokens: input tokens
        :param x: raw features
        :param edge_index: edges
        :return: (E, H, S) Manifold initial representations
        """
        E = self.Euc_init(tokens)
        H = self.Hyp_init(tokens, edge_index)
        S = self.Sph_init(tokens, edge_index)
        return E, H, S


class StructuralBlock(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout):
        super(StructuralBlock, self).__init__()
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        self.Hyp_learner = HyperbolicStructureLearner(self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout)
        self.Sph_learner = SphericalStructureLearner(self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout)
        self.proj = self.proj = nn.Sequential(nn.Linear(3 * out_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, out_dim))

    def forward(self, x_tuple, data):
        """

        :param x_tuple: (x_E, x_H, x_S)
        :param data: Dataset for a graph contains batched sub-graphs and sub-trees
        :return: x_tuple: (x_H, x_S)
        """
        x_E, x_H, x_S = x_tuple
        x_H = self.Hyp_learner(x_H, x_S, data.batch_tree)
        x_S = self.Sph_learner(x_H, x_S, data)

        H_E = self.manifold_H.proju(x_H, x_E)
        S_E = self.manifold_H.proju(x_S, x_E)

        H_E = self.manifold_H.transp0back(x_H, H_E)
        S_E = self.manifold_S.transp0back(x_S, S_E)

        E = torch.cat([x_E, H_E, S_E], dim=-1)
        x_E = self.proj(E)
        return x_E, x_H, x_S