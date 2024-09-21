import torch
from torch_geometric.data import Batch


if __name__ == '__main__':
    import torch
    from torch_geometric.data import Data, Batch

    data1 = Data(x=torch.tensor([[1], [2], [3]]), edge_index=torch.tensor([[0, 1], [1, 2]]))
    data2 = Data(x=torch.tensor([[4], [5]]), edge_index=torch.tensor([[0, 1], [1, 0]]))

    batch = Batch.from_data_list([data1, data2])

    print(batch)
    print(batch.batch)