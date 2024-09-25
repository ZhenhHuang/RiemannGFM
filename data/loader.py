import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import NeighborLoader
from typing import Union, List, Optional, Callable
from torch_geometric.utils import negative_sampling
from data.graph_exacters import hierarchical_exacter


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
                 num_neg_samples: int = None, **kwargs):
        super(ExtractLoader, self).__init__(
            data, num_neighbors, input_nodes, input_time, replace, directed,
            disjoint, temporal_strategy, time_attr, transform,
            transform_sampler_output, is_sorted, filter_per_worker, neighbor_sampler,
            **kwargs
        )
        self.num_neg_samples = num_neg_samples

    def __iter__(self):
        for data in super().__iter__():
            data.neg_edge_index = negative_sampling(data.edge_index, num_neg_samples=self.num_neg_samples)
            tree_list = []
            subset, sub_edge_index = data.n_id, data.edge_index
            for m, seed_node in enumerate(data.n_id[: data.batch_size]):
                tree_edge_index = hierarchical_exacter(subset, sub_edge_index, torch.tensor([m]).long(), flow='source_to_target')
                tree_list.append(Data(edge_index=tree_edge_index, num_nodes=subset.shape[0], seed_node=seed_node))
            batch_tree = Batch.from_data_list(tree_list)
            data.batch_tree = batch_tree
            yield data