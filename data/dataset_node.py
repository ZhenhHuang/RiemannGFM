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
        data.data_dict = {i: graph_exacter(data, i + 1) for i in range(self.configs.n_layers)}
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

        data.data_dict = {i: graph_exacter(data, i + 1) for i in range(self.configs.n_layers)}
        return data

    def len(self) -> int:
        count = 0
        for k, v in self.data.data_dict.items():
            count += len(v)
        return count

    def get(self, idx: int) -> BaseData:
        return self.data


if __name__ == '__main__':
    from utils.config import DotDict
    from dataset_vallina import load_data

    configs = DotDict({"n_layers": 2, "data_name": "KarateClub", "root_path": None})
    # dataset = NodeClsDataset(raw_dataset=load_data(root=configs.root_path,
    #                                                data_name=configs.data_name),
    #                          configs=configs,
    #                          split="train")
    dataset = PretrainingNodeDataset(raw_dataset=load_data(root=configs.root_path,
                                                   data_name=configs.data_name),
                             configs=configs)
