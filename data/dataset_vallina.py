import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.datasets import Amazon, Coauthor, KarateClub
from torch_geometric.transforms import RandomNodeSplit


input_dim_dict = {"KarateClub": 34}


def load_data(root: str, data_name: str, split='public', num_val=0.1, num_test=0.8) -> Dataset:
    if data_name in ["computers", "photo"]:
        dataset = Amazon(root, name=data_name, transform=RandomNodeSplit(num_val=num_val, num_test=num_test))
    elif data_name == "KarateClub":
        dataset = KarateClub(transform=RandomNodeSplit(num_val=0.2, num_test=0.3))
    elif data_name in ["CS", "Physics"]:
        dataset = Coauthor(root, name=data_name, transform=RandomNodeSplit(num_val=0.2, num_test=0.3))
    else:
        raise NotImplementedError

    return dataset