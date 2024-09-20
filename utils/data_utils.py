from torch_geometric.utils import negative_sampling
import torch


def mask_edges(edge_index, neg_edges, val_prop, test_prop):
    n = len(edge_index[0])
    n_val = int(val_prop * n)
    n_test = int(test_prop * n)
    edge_val, edge_test, edge_train = edge_index[:, :n_val], edge_index[:, n_val:n_val + n_test], edge_index[:,
                                                                                                  n_val + n_test:]
    val_edges_neg, test_edges_neg = neg_edges[:, :n_val], neg_edges[:, n_val:n_test + n_val]
    train_edges_neg = torch.concat([neg_edges, val_edges_neg, test_edges_neg], dim=-1)
    return (edge_train, edge_val, edge_test), (train_edges_neg, val_edges_neg, test_edges_neg)