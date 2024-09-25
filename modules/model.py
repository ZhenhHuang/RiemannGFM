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
        self.product = ProductSpace(*[(Euclidean(), embed_dim),
                                      (self.manifold_H, embed_dim),
                                      (self.manifold_S, embed_dim)])
        self.init_block = InitBlock(self.manifold_H, self.manifold_S,
                                    in_dim, hidden_dim, embed_dim, bias,
                                    activation, dropout)
        self.blocks = nn.ModuleList([])
        for i in range(n_layers):
            self.blocks.append(StructuralBlock(self.manifold_H, self.manifold_S,
                                               embed_dim, hidden_dim, embed_dim, dropout))
        self.eps_net = EpsNet(3 * embed_dim, embed_dim, dropout)

    def forward(self, data):
        """

        :param data: Dataset for a graph contains batched sub-graphs and sub-trees.
        :return: z: node product representations
        """
        x = data.x.clone()
        x_E, x_H, x_S = self.init_block(x)  # [N, Hidden]
        for i, block in enumerate(self.blocks):
            x_H, x_S = block((x_H, x_S), data)
        return x_E, x_H, x_S

    def loss(self, x_tuple, data):
        """

        :param x_tuple: (x_E, x_H, x_S)
        :param data:
        :return:
        """
        batch_data = data.batch_data
        # node_labels = batch_data.node_labels
        batch = batch_data.batch
        edge_index = data.edge_index
        neg_edge_index = data.neg_edge_index
        edges = torch.cat([edge_index, neg_edge_index], dim=-1)
        rank = torch.tensor([1.] * edge_index.shape[-1] + [-1.] * neg_edge_index.shape[-1]).to(edges.device)

        x = torch.cat(x_tuple, dim=-1)
        d_p = self.product.dist(x[edges[0]], x[edges[1]])

        # x_rep = x[node_labels]
        # e1 = torch.stack(torch.where(batch[None] == edges[0][:, None]), dim=0)
        # e2 = torch.stack(torch.where(batch[None] == edges[1][:, None]), dim=0)
        #
        # e1_src, e1_dst = e1[0], e1[1]
        # e2_src, e2_dst = e2[0], e2[1]

        x_rep = x[batch]
        x_mid = self.product.Frechet_mean(x_rep, dim=0, sum_idx=batch, keepdim=True)

        # x_src = self.product.Frechet_mean(x_rep[e1_dst], dim=0, sum_idx=e1_src, keepdim=True)
        # x_dst = self.product.Frechet_mean(x_rep[e2_dst], dim=0, sum_idx=e2_src, keepdim=True)
        #
        # d_G = torch.mean(self.product.dist(x_src, x_dst))
        d_G = self.product.dist(x_mid[edges[0]], x_mid[edges[1]])

        eps = self.eps_net(x[edges[0]], x[edges[1]])
        loss = F.relu(rank * (d_p - d_G + eps * d_p)).mean()
        return loss


class InitBlock(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, bias, activation, dropout):
        super(InitBlock, self).__init__()
        self.Euc_init = EuclideanEncoder(in_dim, hidden_dim, out_dim, bias, activation, dropout)
        self.Hyp_init = ManifoldEncoder(manifold_H, out_dim, hidden_dim, out_dim, bias, None, 0.)
        self.Sph_init = ManifoldEncoder(manifold_S, out_dim, hidden_dim, out_dim, bias, None, 0.)

    def forward(self, x):
        """

        :param x: raw features
        :return: (E, H, S) Manifold initial representations
        """
        E = self.Euc_init(x)
        H = self.Hyp_init(E)
        S = self.Sph_init(E)
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
        x_S = self.Sph_learner(x_H, x_S, data.batch_data)
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