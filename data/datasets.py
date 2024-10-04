import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import (Amazon, Coauthor, KarateClub, Planetoid,
                                      GitHub, Airports, Flickr, Reddit, PolBlogs)
from torch_geometric.transforms import RandomNodeSplit, RandomLinkSplit
from ogb.nodeproppred import PygNodePropPredDataset
import os


input_dim_dict = {"KarateClub": 34, "Cora": 1433, "Citeseer": 3703, "PubMed": 500,
                  'ogbn-arxiv': 128, "CS": 6805, "GitHub": 128, "USA": 1190, "computers": 767, "Flickr": 500}
class_num_dict = {"KarateClub": 4, "Cora": 7, "Citeseer": 6, "PubMed": 3, "ogbn-arxiv": 40, "CS": 15,
                  "GitHub": 2, "USA": 4, "computers": 10, "Flickr": 7}


def load_data(root: str, data_name: str, split='public', num_val=0.1, num_test=0.2, num_per_class=None) -> Dataset:
    if data_name in ["computers", "photo"]:
        dataset = Amazon(root, name=data_name, transform=RandomNodeSplit(num_val=num_val, num_test=num_test))
    elif data_name == "KarateClub":
        dataset = KarateClub(transform=RandomNodeSplit(num_val=0.2, num_test=0.3))
    elif data_name in ["CS", "Physics"]:
        dataset = Coauthor(root, name=data_name, transform=RandomNodeSplit(num_val=0.2, num_test=0.3))
    elif data_name in ['Cora', 'Citeseer', 'PubMed']:
        if num_per_class is None:
            num_per_class = 20
        dataset = Planetoid(root, name=data_name, split=split, num_train_per_class=num_per_class)
    elif data_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=data_name, root=root, transform=RandomNodeSplit(num_val=0.2, num_test=0.3))
    elif data_name == 'GitHub':
        if num_per_class is None:
            transform = RandomNodeSplit(num_val=0.1, num_test=0.2)
        else:
            transform = RandomNodeSplit(num_val=0.1, num_test=0.2, num_train_per_class=num_per_class)
        dataset = GitHub(os.path.join(root, "GitHub"), transform=transform)
    elif data_name in ["USA", "Brazil", "Europe"]:
        if num_per_class is None:
            transform = RandomNodeSplit(num_val=0.1, num_test=0.2)
        else:
            transform = RandomNodeSplit(num_val=0.1, num_test=0.2, num_train_per_class=num_per_class)
        dataset = Airports(root, data_name, transform=transform)
    elif data_name == 'Flickr':
        dataset = Flickr(os.path.join(root, "Flickr"))
    elif data_name == 'Reddit':
        dataset = Reddit(root)
    elif data_name == 'PolBlogs':
        dataset = PolBlogs(root, transform=RandomNodeSplit(num_val=0.1, num_test=0.2))
    else:
        raise NotImplementedError
    input_dim_dict[data_name] = dataset.num_features
    class_num_dict[data_name] = dataset.num_classes

    return dataset