import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from modules.layers import EuclideanEncoder, ManifoldEncoder
from modules.basics import HyperbolicStructureLearner, SphericalStructureLearner
from manifolds import Euclidean, Lorentz, Sphere, ProductSpace
from torch_scatter import scatter_mean, scatter_sum


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
        self.proj = nn.Linear(3 * embed_dim, embed_dim)

    def forward(self, data):
        """

        :param data: Dataset for a graph contains batched sub-trees.
        :return: z: node product representations
        """
        x = data.x.clone()
        x_E, x_H, x_S = self.init_block(x, data.edge_index)  # [N, Hidden]
        for i, block in enumerate(self.blocks):
            x_H, x_S = block((x_H, x_S), data)
        H_E = self.manifold_H.proju(x_H, x_E)
        S_E = self.manifold_H.proju(x_S, x_E)

        H_E = self.manifold_H.transp0back(x_H, H_E)
        S_E = self.manifold_S.transp(x_S, self.manifold_S.origin(x_S.shape, device=x_S.device), S_E)
        E = torch.cat([x_E, H_E, S_E], dim=-1)
        x_E = self.proj(E)
        return x_E, x_H, x_S

    def loss(self, x_tuple, data):
        """

        :param x_tuple: (x_E, x_H, x_S)
        :param data:
        :return:
        """

        E, H, S = x_tuple
        E = E[:data.batch_size]
        H = H[:data.batch_size]
        S = S[:data.batch_size]

        H_E = self.manifold_H.logmap0(H)
        S_E = self.manifold_S.logmap0(S)
        loss = self.cal_cl_loss(E, H_E) + self.cal_cl_loss(E, S_E)

        return loss

    def cal_cl_loss(self, x1, x2):
        EPS = 1e-6
        norm1 = x1.norm(dim=-1)
        norm2 = x2.norm(dim=-1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / (torch.einsum('i,j->ij', norm1, norm2) + EPS)
        sim_matrix = torch.exp(sim_matrix / 0.2)
        pos_sim = sim_matrix.diag()
        loss_1 = pos_sim / (sim_matrix.sum(dim=-2) - pos_sim + EPS)
        loss_2 = pos_sim / (sim_matrix.sum(dim=-1) - pos_sim + EPS)

        loss_1 = -torch.log(loss_1).mean()
        loss_2 = -torch.log(loss_2).mean()
        loss = (loss_1 + loss_2) / 2.
        return loss


class InitBlock(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, bias, activation, dropout):
        super(InitBlock, self).__init__()
        self.Euc_init = EuclideanEncoder(in_dim, hidden_dim, out_dim, bias, activation, dropout)
        self.Hyp_init = ManifoldEncoder(manifold_H, out_dim, hidden_dim, out_dim, bias, None, 0.)
        self.Sph_init = ManifoldEncoder(manifold_S, out_dim, hidden_dim, out_dim, bias, None, 0.)

    def forward(self, x, edge_index):
        """

        :param x: raw features
        :param edge_index: edges
        :return: (E, H, S) Manifold initial representations
        """
        E = self.Euc_init(x, edge_index)
        # y = torch.ones_like(E) + torch.arange(E.shape[0], device=E.device).unsqueeze(-1) / E.shape[0]
        y = E
        H = self.Hyp_init(y, edge_index)
        S = self.Sph_init(y, edge_index)
        return E, H, S


class StructuralBlock(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout):
        super(StructuralBlock, self).__init__()
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        self.Hyp_learner = HyperbolicStructureLearner(self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout)
        self.Sph_learner = SphericalStructureLearner(self.manifold_H, self.manifold_S, in_dim, hidden_dim, out_dim, dropout)

    def forward(self, x_tuple, data):
        """

        :param x_tuple: (x_H, x_S)
        :param data: Dataset for a graph contains batched sub-graphs and sub-trees
        :return: x_tuple: (x_H, x_S)
        """
        x_H, x_S = x_tuple
        x_H = self.Hyp_learner(x_H, x_S, data.batch_tree)
        x_S = self.Sph_learner(x_H, x_S, data)
        return x_H, x_S


class EpsNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(EpsNet, self).__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.lin2 = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, y):
        """

        :param x: src nodes
        :param y: dst nodes
        :return:
        """
        x, y = self.lin1(x), self.lin1(y)
        z = torch.concat([x, y], dim=-1)
        z = F.relu(self.lin2(self.drop(z)))
        return z.squeeze(-1)


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