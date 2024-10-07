import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import (Amazon, Coauthor, KarateClub, Planetoid,
                                      GitHub, Airports, Flickr, Reddit, PolBlogs,
                                      WikiCS, )
from torch_geometric.transforms import RandomNodeSplit
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import get_laplacian
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import os

input_dim_dict = {"KarateClub": 34, "Cora": 1433, "Citeseer": 3703, "PubMed": 500,
                  'ogbn-arxiv': 128, "CS": 6805, "GitHub": 128, "USA": 1190, "computers": 767, "Flickr": 500,
                  "WikiCS": 128}
class_num_dict = {"KarateClub": 4, "Cora": 7, "Citeseer": 6, "PubMed": 3, "ogbn-arxiv": 40, "CS": 15,
                  "GitHub": 2, "USA": 4, "computers": 10, "Flickr": 7, "WikiCS": 10}


def load_data(root: str, data_name: str,
              num_val=0.1, num_test=0.2, num_per_class=None) -> Dataset:
    if num_per_class is None:
        transform = RandomNodeSplit(num_val=0.1, num_test=0.2)
    else:
        transform = RandomNodeSplit(num_val=0.1, num_test=0.2, split="test_rest", num_train_per_class=num_per_class)

    if data_name in ["computers", "photo"]:
        dataset = Amazon(root, name=data_name, transform=RandomNodeSplit(num_val=num_val, num_test=num_test))
    elif data_name == "KarateClub":
        dataset = KarateClub(transform=RandomNodeSplit(num_val=0.2, num_test=0.3))
    elif data_name in ["CS", "Physics"]:
        dataset = Coauthor(root, name=data_name, transform=RandomNodeSplit(num_val=0.2, num_test=0.3))
    elif data_name in ['Cora', 'Citeseer', 'PubMed']:
        if num_per_class is None:
            num_per_class = 20
            split = "public"
        else:
            split = "random"
        dataset = Planetoid(root, name=data_name, split=split, num_train_per_class=num_per_class)
    elif data_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=data_name, root=root,
                                         transform=RandomNodeSplit(num_val=0.2, num_test=0.3))
    elif data_name == 'GitHub':
        dataset = GitHub(os.path.join(root, "GitHub"), transform=transform)
    elif data_name in ["USA", "Brazil", "Europe"]:
        dataset = Airports(root, data_name, transform=transform)
    elif data_name == 'Flickr':
        dataset = Flickr(os.path.join(root, data_name))
    elif data_name == 'Reddit':
        dataset = Reddit(root)
    elif data_name == 'PolBlogs':
        dataset = PolBlogs(root, transform=RandomNodeSplit(num_val=0.1, num_test=0.2))
    elif data_name == "WikiCS":
        dataset = WikiCS(os.path.join(root, data_name), transform=transform)
    else:
        raise NotImplementedError
    input_dim_dict[data_name] = dataset.num_features
    class_num_dict[data_name] = dataset.num_classes
    return dataset


def get_eigen_tokens(data, embed_dim, device):
    n = data.num_nodes
    edge_index, edge_weight = get_laplacian(data.edge_index, normalization="sym", num_nodes=n)
    row, col = edge_index[0].numpy(), edge_index[1].numpy()
    L = csr_matrix((edge_weight.numpy(), (row, col)), shape=(n, n))
    _, eigvecs = eigs(L, k=embed_dim, which='SM')
    eigvecs = torch.tensor(eigvecs.real).to(device)

    def tokens(idx):
        return eigvecs[idx]

    return tokens