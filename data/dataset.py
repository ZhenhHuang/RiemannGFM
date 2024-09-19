import torch
from torch_geometric.data import Batch, Dataset, InMemoryDataset
from torch_geometric.datasets import KarateClub
from torch_geometric.typing import OptTensor


if __name__ == '__main__':
    from graph_exacters import graph_exacter
    dataset = KarateClub()
    data_list = graph_exacter(dataset.get(0), k_hop=2)
    Dataset()