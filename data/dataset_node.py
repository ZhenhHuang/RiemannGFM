from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import BaseData
from graph_exacters import graph_exacter
import torch


class PretrainingNodeDataset(Dataset):
    """
    A dataset for pre-training that contains only one graph.
    """
    def __init__(self, raw_dataset: Dataset, configs):
        super(PretrainingNodeDataset, self).__init__()
        self.raw_dataset = raw_dataset
        self.configs = configs
        self.data = self._extract()

    def _extract(self):
        data = self.raw_dataset.get(0).clone()
        data.data_dict = {i: graph_exacter(data, i + 1) for i in self.configs.n_layers}
        return data

    def len(self) -> int:
        count = 0
        for k, v in self.data.data_dict.items():
            count += len(v)
        return count

    def get(self, idx: int) -> BaseData:
        return self.data


class NodeClsDataset(Dataset):
    def __init__(self, raw_dataset: Dataset, configs, split: str = "train"):
        super(NodeClsDataset, self).__init__()
        self.raw_dataset = raw_dataset
        self.configs = configs
        self.data = self._extract(split)

    def _extract(self, split):
        data = self.raw_dataset.get(0).clone()
        if split == "train":
            mask = data.train_mask
        elif split == "val":
            mask = data.val_mask
        else:
            mask = data.test_mask
        data.x = data.x[mask]
        data.y = data.y[mask]
        edge_index = data.edge_index
        data.edge_index = data.edge_index[:, mask[edge_index[0]]]
        data.data_dict = {i: graph_exacter(data, i + 1) for i in self.configs.n_layers}
        return data

    def len(self) -> int:
        count = 0
        for k, v in self.data.data_dict.items():
            count += len(v)
        return count

    def get(self, idx: int) -> BaseData:
        return self.data