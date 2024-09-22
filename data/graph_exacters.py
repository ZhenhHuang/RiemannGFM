import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from queue import Queue
from collections import defaultdict


def graph_exacter(data, k_hop):
    sub_graphs = []
    num_nodes, edge_index = data.x.shape[0], data.edge_index
    for node in range(num_nodes):
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(node, k_hop, edge_index,
                                                                    num_nodes=num_nodes, relabel_nodes=True)
        tree_dict, k_hop_tree = hierarchical_exacter(subset, sub_edge_index, mapping)
        sub_graphs.append(Data(edge_index=sub_edge_index, num_nodes=len(subset),
                               subset=subset, central_node=node, tree_dict=tree_dict))
    return sub_graphs


def hierarchical_exacter(subset, edge_index, mapping, flow: str = 'source_to_target'):
    """

    :param flow:
    :param subset: node set of sub-graph
    :param edge_index: relabeled edge set of sub-graph
    :param mapping: the new location of central node
    :return: alignment_dict, a directed tree graph: (edge_index)
    """
    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index
    tree_edges = []
    align_dict = defaultdict(list)
    visited = subset.new_empty(subset.size(0), dtype=torch.bool)
    visited.fill_(False)
    que, h_que = Queue(), Queue()
    que.put(mapping)
    h_que.put(1)
    visited[mapping] = True
    while not que.empty():
        if (visited == True).all():
            break
        node = que.get()
        h = h_que.get()
        idx = torch.where(node == row)[0]
        source = col[idx]
        idx = torch.where(visited[source] == False)[0]
        child = source[idx]
        visited[child] = True
        if flow == 'target_to_source':
            edges = torch.cartesian_prod(node, child)
        else:
            edges = torch.cartesian_prod(child, node)
        tree_edges.append(edges)
        align_dict[h].append(edges)
        for ch in child:
            que.put(ch.reshape(-1))
            h_que.put(h + 1)

    if len(tree_edges):
        tree_edges = torch.sort(torch.cat(tree_edges, dim=0).t(), dim=-1)[0]
    for k, v in align_dict.items():
        align_dict[k] = torch.sort(torch.cat(v, dim=0).t(), dim=-1)[0]
    return align_dict, tree_edges


# if __name__ == '__main__':
#     from torch_geometric.utils import to_undirected
#     edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
#                                [2, 2, 4, 4, 6, 6]])
#     edge_index = to_undirected(edge_index)
#     subset, edge_index, mapping, edge_mask = k_hop_subgraph(
#     6, 2, edge_index, relabel_nodes=True)
#     t_dict, tree = hierarchical_exacter(subset, edge_index, mapping)
