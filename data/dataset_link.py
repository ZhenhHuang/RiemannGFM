from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.utils import negative_sampling
from graph_exacters import graph_exacter
from utils.data_utils import mask_edges
import torch


class LinkPredDataset(Dataset):
    """
    A dataset for Link Prediction that contains only one graph.
    """
    def __init__(self, raw_dataset: Dataset, configs, split: str = "train"):
        super(LinkPredDataset, self).__init__()
        self.raw_dataset = raw_dataset
        self.configs = configs
        self._mask_edges()
        self.data = self._extract(split)

    def _mask_edges(self):
        data = self.raw_dataset[0].clone()
        edge_index = data.edge_index
        neg_edge_index = negative_sampling(edge_index)
        pos_edges, neg_edges = mask_edges(edge_index, neg_edge_index, val_prop=0.05, test_prop=0.1)
        edge_train, edge_val, edge_test = pos_edges
        train_edges_neg, val_edges_neg, test_edges_neg = neg_edges
        self.pos_edge_train = edge_train
        self.pos_edge_val = edge_val
        self.pos_edge_test = edge_test
        self.neg_edge_train = train_edges_neg
        self.neg_edge_val = val_edges_neg
        self.neg_edge_test = test_edges_neg

    def _extract(self, split):
        data = self.raw_dataset[0].clone()
        if split == "train":
            data.edge_index = self.pos_edge_train
            data.neg_edge_index = self.neg_edge_train
        elif split == "val":
            data.edge_index = self.pos_edge_val
            data.neg_edge_index = self.neg_edge_val
        else:
            data.edge_index = self.pos_edge_test
            data.neg_edge_index = self.neg_edge_test

        data.data_dict = {i: graph_exacter(data, i + 1) for i in range(self.configs.n_layers)}
        return data

    def len(self) -> int:
        return 1

    def get(self, idx: int) -> BaseData:
        return self.data


if __name__ == '__main__':
    from utils.config import DotDict
    from dataset_vallina import load_data
    from torch_geometric.loader import DataLoader

    configs = DotDict({"n_layers": 2, "data_name": "KarateClub", "root_path": None})
    dataset = LinkPredDataset(raw_dataset=load_data(root=configs.root_path,
                                                   data_name=configs.data_name),
                             configs=configs,
                             split="train")
    loader = DataLoader(dataset, batch_size=1)
    for data in loader:
        print(data)