from .dataset_vallina import load_data, input_dim_dict, class_num_dict
from .dataset_node import PretrainingNodeDataset, NodeClsDataset
from .dataset_link import LinkPredDataset


__all__ = ['load_data', 'input_dim_dict', 'class_num_dict',
           'PretrainingNodeDataset', 'NodeClsDataset',
           'LinkPredDataset']