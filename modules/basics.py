import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from modules.layers import ConstCurveLinear
from torch_scatter import scatter_sum, scatter_softmax


class HyperbolicStructureLearner(nn.Module):
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(HyperbolicStructureLearner, self).__init__()
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        self.tree_agg = CrossManifoldAttention(manifold_S, manifold_H, in_dim, hidden_dim, out_dim, dropout)

    def forward(self, x_H, x_S, batch_tree):
        """
        Local Attention based on BFS tree structure inherit from a sub-graph.
        :param x_H: Hyperbolic representation of nodes
        :param batch_tree: a batch graph with tree-graphs from one graph.
        :return: New Hyperbolic representation of nodes.
        """
        num_seeds = len(batch_tree)
        node_labels = torch.arange(x_H.shape[0], device=x_H.device).repeat(num_seeds)
        x = x_H[node_labels]
        att_index = batch_tree.edge_index
        x = self.tree_agg(x_S[node_labels], x, x, edge_index=att_index)

        x_extend = torch.concat([x, x_H], dim=0)
        label_extend = torch.cat(
            [node_labels, torch.arange(x_H.shape[0], device=x_H.device)],
            dim=0)
        z_H = self.manifold_H.Frechet_mean(x_extend, keepdim=True, sum_idx=label_extend)
        return z_H


class SphericalStructureLearner(nn.Module):
    """
    in_dim = out_dim
    """
    def __init__(self, manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(SphericalStructureLearner, self).__init__()
        self.manifold_H = manifold_H
        self.manifold_S = manifold_S
        self.attention_subset = CrossManifoldAttention(manifold_H, manifold_S, in_dim, hidden_dim, out_dim, dropout)
        self.res_lin = nn.Linear(out_dim, out_dim)

    def forward(self, x_H, x_S, data):
        """

        :param x_H: Hyperbolic representation of nodes
        :param x_S: Sphere representation of nodes
        :param data: a graph or mini-batched graph
        :return: New sphere representation of nodes.
        """
        att_index = data.edge_index
        x = self.attention_subset(x_H, x_S, x_S, edge_index=att_index)
        z_S = self.manifold_S.expmap(x, self.manifold_S.proju(x, self.res_lin(x_S)))
        return z_S


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

        out = scatter_sum(score.unsqueeze(1) * v[dst], agg_index, dim=0, out=self.manifold_k.origin(q.shape, device=q.device))

        denorm = self.manifold_k.inner(None, out, keepdim=True)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        out = 1. / self.manifold_k.k.sqrt() * out / denorm
        out = self.proj(out)
        return out