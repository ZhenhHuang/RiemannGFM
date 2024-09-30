import torch
from torch_geometric.utils import to_undirected


def label2node(data, num_classes):
    y_ext = data.y + data.num_nodes
    label_edges = torch.stack([torch.arange(data.num_nodes), y_ext], dim=0)
    label_edges = to_undirected(label_edges)
    label_nodes = torch.rand(num_classes, data.num_features)
    data.x = torch.cat([data.x, label_nodes], dim=0)
    data.edge_index = torch.cat([data.edge_index, label_edges], dim=-1)
    return data