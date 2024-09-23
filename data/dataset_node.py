from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.data.data import BaseData
from data.graph_exacters import graph_exacter, hierarchical_exacter
from torch_geometric.utils import negative_sampling
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
        data = self.raw_dataset[0].clone()
        neg_edge_index = negative_sampling(data.edge_index)
        data.neg_edge_index = neg_edge_index
        data = graph_exacter(data, self.configs.hops)
        return data

    def len(self) -> int:
        return 1

    def get(self, idx: int) -> BaseData:
        return self.data


class InductiveNodeClsDataset(Dataset):
    """
    An inductive dataset for node classification that contains only one graph.
    """
    def __init__(self, raw_dataset: Dataset, configs, split: str = "train"):
        super(InductiveNodeClsDataset, self).__init__()
        self.raw_dataset = raw_dataset
        self.configs = configs
        self.data = self._extract(split)

    def _extract(self, split):
        data = self.raw_dataset[0].clone()
        if split == "train":
            mask = data.train_mask
        elif split == "val":
            mask = data.val_mask
        else:
            mask = data.test_mask
        num_nodes = data.x.shape[0]
        data.x = data.x[mask]
        data.y = data.y[mask]
        edge_index = data.edge_index.clone()
        edge_index = data.edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]
        # relabel edge_index
        row = edge_index[1]
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[mask] = torch.arange(mask.sum(), device=row.device)
        data.edge_index = node_idx[edge_index]

        data = graph_exacter(data, self.configs.hops)
        return data

    def len(self) -> int:
        return 1

    def get(self, idx: int) -> BaseData:
        return self.data


class TransductiveNodeClsDataset(Dataset):
    """
    An inductive dataset for node classification that contains only one graph.
    """
    def __init__(self, raw_dataset: Dataset, configs, split: str = "train"):
        super(TransductiveNodeClsDataset, self).__init__()
        self.raw_dataset = raw_dataset
        self.configs = configs
        self.data = self._extract(split)

    def _extract(self, split):
        data = self.raw_dataset[0].clone()
        if split == "train":
            mask = data.train_mask
        elif split == "val":
            mask = data.val_mask
        else:
            mask = data.test_mask
        data.mask = mask
        data = graph_exacter(data, self.configs.hops)
        return data

    def len(self) -> int:
        return 1

    def get(self, idx: int) -> BaseData:
        return self.data