import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from typing import Union, List, Optional, Callable
from torch_geometric.utils import negative_sampling, is_undirected
from data.graph_exacters import graph_exacter


class ExtractLoader(NeighborLoader):
    def __init__(self, data,
                 num_neighbors,
                 input_nodes=None,
                 input_time=None,
                 replace: bool = False,
                 directed: bool = True,
                 disjoint: bool = False,
                 temporal_strategy: str = 'uniform',
                 time_attr: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 transform_sampler_output: Optional[Callable] = None,
                 is_sorted: bool = False,
                 filter_per_worker: bool = False,
                 neighbor_sampler=None,
                 num_neg_samples: int = None, hops: List[int] = None, **kwargs):
        super(ExtractLoader, self).__init__(
            data, num_neighbors, input_nodes, input_time, replace, directed,
            disjoint, temporal_strategy, time_attr, transform,
            transform_sampler_output, is_sorted, filter_per_worker, neighbor_sampler,
            **kwargs
        )
        self.num_neg_samples = num_neg_samples
        self.hops = hops

    def handle_subgraph(self, data):
        neg_edge_index = negative_sampling(data.edge_index, num_neg_samples=self.num_neg_samples)
        data.neg_edge_index = neg_edge_index
        data.n_id = torch.arange(data.num_nodes)
        data = graph_exacter(data, self.hops)
        return data

    def __iter__(self):
        for batch_data in super().__iter__():
            batch_data = self.handle_subgraph(batch_data)
            yield batch_data
