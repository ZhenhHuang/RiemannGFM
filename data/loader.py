import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from typing import Union, List, Optional, Callable
from torch_geometric.utils import add_self_loops, to_networkx, from_networkx
from data.graph_exacters import hierarchical_exacter
from collections import OrderedDict
import networkx as nx
from utils.data_utils import label2node


class ExtractNodeLoader(NeighborLoader):
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
                 capacity: int = 1000, K_shot: int = None, num_classes=None, **kwargs):
        super(ExtractNodeLoader, self).__init__(
            data, num_neighbors, input_nodes, input_time, replace, directed,
            disjoint, temporal_strategy, time_attr, transform,
            transform_sampler_output, is_sorted, filter_per_worker, neighbor_sampler,
            **kwargs
        )
        self.cache = LRUCache(capacity=capacity)
        #   TODO: K_shot
        self.num_classes = num_classes

    def __iter__(self):
        for key, data in enumerate(super().__iter__()):
            if self.num_classes is not None:
                data = label2node(data, self.num_classes)
                data.n_id = torch.cat(
                    [data.n_id, torch.arange(self.data.num_nodes, self.data.num_nodes + self.num_classes)], dim=0)
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            if key in self.cache:
                data = self.cache.get(key)
            else:
                tree_list = []
                subset, sub_edge_index = data.n_id, data.edge_index
                G = to_networkx(Data(edge_index=sub_edge_index, num_nodes=subset.shape[0]))
                for m, seed_node in enumerate(data.n_id[:data.batch_size]):
                    sorted_edges = sorted(list(nx.bfs_tree(G, m).edges()))
                    tG = nx.Graph()
                    tG.add_edges_from(sorted_edges)
                    tree_edge_index = from_networkx(tG).edge_index
                    del tG
                    # tree_edge_index = hierarchical_exacter(subset, sub_edge_index, torch.tensor([m]).long(), flow='source_to_target')
                    tree_list.append(Data(edge_index=tree_edge_index, num_nodes=subset.shape[0], seed_node=seed_node))
                batch_tree = Batch.from_data_list(tree_list)
                data.batch_tree = batch_tree
                self.cache.put(key, data)
            yield data

    def clear_cache(self):
        self.cache.clear()


class ExtractLinkLoader(LinkNeighborLoader):
    def __init__(self, data,
                num_neighbors,
                edge_label_index=None,
                edge_label=None,
                edge_label_time=None,
                replace: bool = False,
                directed: bool = True,
                disjoint: bool = False,
                temporal_strategy: str = 'uniform',
                neg_sampling=None,
                neg_sampling_ratio: Optional[Union[int, float]] = None,
                time_attr: Optional[str] = None,
                transform: Optional[Callable] = None,
                transform_sampler_output: Optional[Callable] = None,
                is_sorted: bool = False,
                filter_per_worker: bool = False,
                neighbor_sampler=None,
                capacity: int = 1000, **kwargs):
        super(ExtractLinkLoader, self).__init__(
            data, num_neighbors, edge_label_index, edge_label, edge_label_time, replace, directed, disjoint, temporal_strategy,
            neg_sampling, neg_sampling_ratio, time_attr, transform, transform_sampler_output, is_sorted,
            filter_per_worker, neighbor_sampler, **kwargs
        )
        self.cache = LRUCache(capacity=capacity)

    def __iter__(self):
        for key, data in enumerate(super().__iter__()):
            data.batch_size = self.batch_size
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
            if key in self.cache:
                data = self.cache.get(key)
            else:
                tree_list = []
                subset, sub_edge_index = data.n_id, data.edge_index
                G = to_networkx(Data(edge_index=sub_edge_index, num_nodes=subset.shape[0]))
                for m, seed_node in enumerate(data.n_id[: data.batch_size]):
                    sorted_edges = sorted(list(nx.bfs_tree(G, m).edges()))
                    tG = nx.Graph()
                    tG.add_edges_from(sorted_edges)
                    tree_edge_index = from_networkx(tG).edge_index
                    del tG
                    # tree_edge_index = hierarchical_exacter(subset, sub_edge_index, torch.tensor([m]).long(), flow='source_to_target')
                    tree_list.append(Data(edge_index=tree_edge_index, num_nodes=subset.shape[0], seed_node=seed_node))
                batch_tree = Batch.from_data_list(tree_list)
                data.batch_tree = batch_tree
                self.cache.put(key, data)
            yield data

    def clear_cache(self):
        self.cache.clear()


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def __contains__(self, item):
        return item in self.cache