import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from layers import ConstCurveLinear, ConstCurveAgg
from manifolds import Sphere, Lorentz
from torch_scatter import scatter_sum, scatter_softmax


class HyperbolicStructureLearner(nn.Module):
    def __init__(self, manifold, in_dim):
        super(HyperbolicStructureLearner, self).__init__()
        assert isinstance(manifold, Lorentz), "The manifold must be a Hyperboloid!"
        self.manifold = manifold
        self.tree_agg = ConstCurveAgg(manifold, in_dim)
        self.attention_agg = ManifoldAttention(manifold, in_dim)

    def forward(self, x_H, data_list):
        z_dict = {i: [x_H[i]] for i in range(len(data_list))}
        for data in data_list:
            subset, tree_dict = data.subset, data.tree_dict
            x_H_sub = x_H[subset].clone()
            k_hop = max(list(tree_dict.keys()))
            for k in range(k_hop, 0, -1):
                sub_edges = tree_dict[k]
                dst = sub_edges[1]
                x_out = self.tree_agg(x_H_sub, sub_edges)
                idx = torch.unique(dst, sorted=True)
                for n in idx:
                    z_dict[subset[n].item()].append(x_out[n])
                x_H_sub[idx] = x_out[idx]
        results = []
        for k, v_l in z_dict.items():
            zs = torch.stack(v_l, dim=0)
            results.append(self.attention_agg(zs.narrow(0, 0, 1), zs, zs))
        z_S = torch.stack(results, dim=0)
        return z_S


class SphericalStructureLearner(nn.Module):
    def __init__(self, manifold, in_dim):
        super(SphericalStructureLearner, self).__init__()
        assert isinstance(manifold, Sphere), "The manifold must be a Sphere!"
        self.manifold = manifold
        self.attention_subset = ManifoldAttention(manifold, in_dim)
        self.attention_agg = ManifoldAttention(manifold, in_dim)

    # def forward(self, x_S, data_list):
    #     """
    #
    #     :param x_S: Spherical representation
    #     :param data_list:
    #     :return:
    #     """
    #     z_dict = {i: [x_S[i]] for i in range(len(data_list))}
    #     for data in data_list:
    #         subset, sub_edge_index = data.subset, data.edge_index
    #         x_S_sub = x_S[subset]
    #         att_out = self.attention_subset(x_S_sub)
    #         for n in subset:
    #             z_dict[n.item()].append(att_out[torch.where(n == subset)[0].item()])
    #     results = []
    #     for k, v_l in z_dict.items():
    #         zs = torch.stack(v_l, dim=0)
    #         results.append(self.attention_agg(zs.narrow(0, 0, 1), zs, zs))
    #     z_S = torch.stack(results, dim=0)
    #     return z_S

    def forward(self, x_S, batch_data):
        """

        :param x_S: Sphere representation of nodes
        :param batch_data: a batch graph with sub-graphs from one graph.
        :return: New sphere representation of nodes.
        """
        node_label = batch_data.node_label
        batch = batch_data.batch
        x = x_S[node_label]
        att_index = torch.cat(torch.where(batch[None] == batch[:, None]), dim=0)
        x = self.attention_subset(x, edge_index=att_index)

        x_extend = torch.concat([x, x_S], dim=0)
        label_extend = torch.cat(
            [node_label, torch.arange(x_S.shape[0], device=x_S.device)],
            dim=0)
        att_index = torch.cat(
            torch.where(label_extend[None] == label_extend[:, None]),
            dim=0)
        z_S = self.attention_agg(x_extend, edge_index=att_index)
        return z_S


class ManifoldAttention(nn.Module):
    def __init__(self, manifold, in_dim):
        super(ManifoldAttention, self).__init__()
        self.manifold = manifold
        self.q_lin = ConstCurveLinear(manifold, in_dim, in_dim, bias=False)
        self.k_lin = ConstCurveLinear(manifold, in_dim, in_dim, bias=False)
        self.v_lin = ConstCurveLinear(manifold, in_dim, in_dim, bias=False)

    def forward(self, x_q, x_k=None, x_v=None, edge_index=None):
        if x_k is None:
            x_k = x_q.clone()
        if x_v is None:
            x_v = x_q.clone()

        q = self.q_lin(x_q)
        k = self.k_lin(x_k)
        v = self.v_lin(x_v)
        if edge_index is None:
            out = self.global_attention(q, k, v)
        else:
            src, dst = edge_index[0], edge_index[1]
            score = self.manifold.inner(None, q[src], k[dst])
            score = scatter_softmax(score, src)
            out = scatter_sum(score * v[dst], src)
            denorm = self.manifold.inner(None, out, keepdim=keepdim)
            denorm = denorm.abs().clamp_min(1e-8).sqrt()
            out = 1. / self.manifold.k.sqrt() * out / denorm
        return out

    def global_attention(self, q, k, v):
        scores = self.manifold.cinner(q, k)
        A = torch.softmax(scores, dim=-1)
        out = self.manifold.Frechet_mean(v.unsqueeze(0), A.unsqueeze(-1), dim=-2, keepdim=True).squeeze()
        return out


# if __name__ == '__main__':
#     from data.graph_exacters import graph_exacter
#     from torch_geometric.datasets import KarateClub
#     from layers import ManifoldEncoder
#     dataset = KarateClub()
#     # manifold = Sphere()
#     manifold = Lorentz()
#     data_list = graph_exacter(dataset.get(0), k_hop=2)
#     encoder = ManifoldEncoder(manifold, 34, 5)
#     x = encoder(dataset.get(0).x)
#     # learner = SphericalStructureLearner(manifold, 6)
#     learner = HyperbolicStructureLearner(manifold, 6)
#     y = learner(x, data_list)
#     print(manifold.check_point_on_manifold(y))



