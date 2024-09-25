import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from modules.layers import ConstCurveLinear, ConstCurveAgg
from manifolds import Sphere, Lorentz
from torch_scatter import scatter_sum, scatter_softmax
from torch_geometric.utils import dropout_edge


class HyperbolicStructureLearner(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(HyperbolicStructureLearner, self).__init__()
        # assert isinstance(manifold, Lorentz), "The manifold must be a Hyperboloid!"
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        self.tree_agg = CrossManifoldAttention(manifold_S, manifold_H, in_dim, hidden_dim, out_dim, dropout)
        # self.attention_agg = ManifoldAttention(manifold_H, out_dim, hidden_dim, out_dim, dropout)
        # self.res_lin = nn.Linear(out_dim, out_dim)

    def forward(self, x_H, x_S, batch_tree):
        """
        Local Attention based on BFS tree structure inherit from a sub-graph.
        :param x_H: Hyperbolic representation of nodes
        :param batch_tree: a batch graph with tree-graphs from one graph.
        :return: New Hyperbolic representation of nodes.
        """
        node_labels = batch_tree.node_labels
        x = x_H[node_labels]
        # x_res = x.clone()
        att_index = batch_tree.edge_index
        x = self.tree_agg(x_S[node_labels], x, x, edge_index=att_index)
        # x = self.manifold_H.expmap(x, self.res_lin(x_res))

        x_extend = torch.concat([x, x_H], dim=0)
        # label_extend = torch.cat(
        #     [node_labels, torch.arange(x_H.shape[0], device=x_H.device)],
        #     dim=0)
        # att_index = torch.stack(
        #     torch.where(label_extend[None] == label_extend[:, None]),
        #     dim=0)
        # agg_index = label_extend[att_index]
        # z_H = self.attention_agg(x_extend, edge_index=att_index, agg_index=agg_index[0])
        batch = batch_tree.batch
        batch_extend = torch.cat(
            [batch, torch.arange(x_H.shape[0], device=x_H.device)],
            dim=0)
        z_H = self.manifold_H.Frechet_mean(x_extend, keepdim=True, sum_idx=batch_extend)
        return z_H


class SphericalStructureLearner(nn.Module):
    """
    in_dim = out_dim
    """
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(SphericalStructureLearner, self).__init__()
        # assert isinstance(manifold, Sphere), "The manifold must be a Sphere!"
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        self.attention_subset = CrossManifoldAttention(manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout)
        # self.attention_agg = ManifoldAttention(manifold_S, out_dim, hidden_dim, out_dim, dropout)
        self.res_lin = nn.Linear(out_dim, out_dim)

    def forward(self, x_H, x_S, batch_data):
        """

        :param x_H: Hyperbolic representation of nodes
        :param x_S: Sphere representation of nodes
        :param batch_data: a batch graph with sub-graphs from one graph.
        :return: New sphere representation of nodes.
        """
        node_labels = batch_data.node_labels
        batch = batch_data.batch
        x = x_S[node_labels]
        x_res = x.clone()
        # att_index = torch.stack(torch.where(batch[None] == batch[:, None]), dim=0)
        att_index = batch_data.edge_index
        x = self.attention_subset(x_H[node_labels], x, x, edge_index=att_index)
        x = self.manifold_S.expmap(x, self.manifold_S.proju(x, self.res_lin(x_res)))

        x_extend = torch.concat([x, x_S], dim=0)
        # label_extend = torch.cat(
        #     [node_labels, torch.arange(x_S.shape[0], device=x_S.device)],
        #     dim=0)
        # att_index = torch.stack(
        #     torch.where(label_extend[None] == label_extend[:, None]),
        #     dim=0)
        # agg_index = label_extend[att_index]
        # z_S = self.attention_agg(x_extend, edge_index=att_index, agg_index=agg_index[0])
        batch_extend = torch.cat(
            [batch, torch.arange(x_S.shape[0], device=x_S.device)],
            dim=0)
        z_S = self.manifold_S.Frechet_mean(x_extend, keepdim=True, sum_idx=batch_extend)
        return z_S


class ManifoldAttention(nn.Module):
    def __init__(self, manifold, in_dim, hidden_dim, out_dim, dropout):
        super(ManifoldAttention, self).__init__()
        self.manifold = manifold
        self.q_lin = ConstCurveLinear(manifold, in_dim, hidden_dim, bias=False, dropout=dropout)
        self.k_lin = ConstCurveLinear(manifold, in_dim, hidden_dim, bias=False, dropout=dropout)
        self.v_lin = ConstCurveLinear(manifold, in_dim, hidden_dim, bias=False, dropout=dropout)
        self.proj = ConstCurveLinear(manifold, hidden_dim, out_dim, bias=False, dropout=dropout)
        self.scalar_map = nn.Sequential(
            nn.Linear(2 * hidden_dim, 1, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x_q, x_k=None, x_v=None, edge_index=None, agg_index=None):
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
            agg_index = agg_index if agg_index is not None else src
            qk = torch.cat([q[src], k[dst]], dim=-1)
            score = self.scalar_map(qk).squeeze(-1)
            score = scatter_softmax(score, src, dim=-1)
            out = scatter_sum(score.unsqueeze(1) * v[dst], agg_index, dim=0)
            denorm = self.manifold.inner(None, out, keepdim=True)
            denorm = denorm.abs().clamp_min(1e-8).sqrt()
            out = 1. / self.manifold.k.sqrt() * out / denorm
        out = self.proj(out)
        return out

    def global_attention(self, q, k, v):
        scores = self.manifold.cinner(q, k)
        A = torch.softmax(scores, dim=-1)
        out = self.manifold.Frechet_mean(v.unsqueeze(0), A.unsqueeze(-1), dim=-2, keepdim=True).squeeze()
        return out


class CrossManifoldAttention(nn.Module):
    def __init__(self, manifold_q, manifold_k, in_dim, hidden_dim, out_dim, dropout):
        super(CrossManifoldAttention, self).__init__()
        self.manifold_q = manifold_q
        self.manifold_k = manifold_k
        self.q_lin = ConstCurveLinear(manifold_q, in_dim, hidden_dim, bias=False, dropout=dropout)
        self.k_lin = ConstCurveLinear(manifold_k, in_dim, hidden_dim, bias=False, dropout=dropout)
        self.v_lin = ConstCurveLinear(manifold_k, in_dim, hidden_dim, bias=False, dropout=dropout)
        self.scalar_map = nn.Sequential(
            nn.Linear(2 * hidden_dim, 1, bias=False),
            nn.LeakyReLU()
        )
        self.proj = ConstCurveLinear(manifold_k, hidden_dim, out_dim, bias=False, dropout=dropout)

    def forward(self, x_q, x_k, x_v, edge_index, agg_index=None):
        q = self.q_lin(x_q)
        k = self.k_lin(x_k)
        v = self.v_lin(x_v)
        src, dst = edge_index[0], edge_index[1]
        agg_index = agg_index if agg_index is not None else src
        qk = torch.cat([q[src], k[dst]], dim=-1)
        score = self.scalar_map(qk).squeeze(-1)
        score = scatter_softmax(score, src, dim=-1)
        out = scatter_sum(score.unsqueeze(1) * v[dst], agg_index, dim=0)
        denorm = self.manifold_k.inner(None, out, keepdim=True)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        out = 1. / self.manifold_k.k.sqrt() * out / denorm
        out = self.proj(out)
        return out


# if __name__ == '__main__':
#     from data.graph_exacters import graph_exacter, hierarchical_exacter
#     from torch_geometric.datasets import KarateClub
#     from layers import ManifoldEncoder
#     from torch_geometric.utils import k_hop_subgraph
#     from torch_geometric.data import Data, Batch
#     dataset = KarateClub()
#     data = dataset[0]
#     edge_index = data.edge_index
#     node_labels = []
#     data_list = []
#     tree_list = []
#     for node in range(data.num_nodes):
#         subset, sub_edge_index, mapping, _ = k_hop_subgraph(node, 2, edge_index,
#                                                                 num_nodes=data.num_nodes, relabel_nodes=True)
#         _, tree_edge_index = hierarchical_exacter(subset, sub_edge_index, mapping, flow='target_to_source')
#         node_labels.append(subset)
#         data_list.append(Data(edge_index=sub_edge_index, num_nodes=subset.shape[0], seed_node=node))
#         tree_list.append(Data(edge_index=tree_edge_index, num_nodes=subset.shape[0], seed_node=node))
#     node_labels = torch.cat(node_labels, dim=0)
#
#     # batch_data = Batch.from_data_list(data_list)
#     # batch_data.node_labels = node_labels
#     # manifold = Sphere()
#     # learner = SphericalStructureLearner(manifold, 5)
#
#     batch_tree = Batch.from_data_list(tree_list)
#     batch_tree.node_labels = node_labels
#     manifold = Lorentz()
#     learner = HyperbolicStructureLearner(manifold, 5)
#
#     encoder = ManifoldEncoder(manifold, 34, 5)
#     x = encoder(data.x)
#
#     # y = learner(x, batch_data)
#     y = learner(x, batch_tree)
#     print(manifold.check_point_on_manifold(y))