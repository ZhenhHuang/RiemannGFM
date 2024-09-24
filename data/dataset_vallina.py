import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import Amazon, Coauthor, KarateClub, Planetoid
from torch_geometric.transforms import RandomNodeSplit
from ogb.nodeproppred import PygNodePropPredDataset


input_dim_dict = {"KarateClub": 34, "Cora": 1433, "Citeseer": 3703, "PubMed": 500,
                  'ogbn-arxiv': 128}
class_num_dict = {"KarateClub": 4, "Cora": 7, "Citeseer": 6, "PubMed": 3, "ogbn-arxiv": 40}


def load_data(root: str, data_name: str, split='public', num_val=0.1, num_test=0.8) -> Dataset:
    if data_name in ["computers", "photo"]:
        dataset = Amazon(root, name=data_name, transform=RandomNodeSplit(num_val=num_val, num_test=num_test))
    elif data_name == "KarateClub":
        dataset = KarateClub(transform=RandomNodeSplit(num_val=0.2, num_test=0.3))
    elif data_name in ["CS", "Physics"]:
        dataset = Coauthor(root, name=data_name, transform=RandomNodeSplit(num_val=0.2, num_test=0.3))
    elif data_name in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(root, name=data_name)
    elif data_name == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(name=data_name, root=root)
    else:
        raise NotImplementedError

    return dataset