import torch
import numpy as np
import os
import random
import argparse
from exp import *
from utils.logger import create_logger

seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='Geometric Graph Foundation Model')

"""Dataset settings"""
parser.add_argument('--task', type=str, default='NC',
                    choices=['NC', 'LP', 'GC', 'Pretrain'])
parser.add_argument('--dataset', type=str, default='Cora',
                    help="['computers', 'photo', 'KarateClub', 'CS', 'Physics']")
parser.add_argument('--pretrain_dataset', nargs="+", type=str, default=['Cora'], help="[ogbn-arxiv, PubMed]")
parser.add_argument('--root_path', type=str, default='D:\datasets\Graphs')
parser.add_argument('--num_neighbors', type=int, nargs="+", default=[20, 10],
                    help="Number of neighbors of data_loaders")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--capacity', type=int, default=1000, help="Capacity of Cache for dataloader")

"""Checkpoints and logger"""
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--pretrained_model_path', type=str)  # necessary
parser.add_argument('--task_model_path', type=str)  # necessary
parser.add_argument('--log_dir', type=str, default='./logs/')
parser.add_argument('--log_name', type=str)  # necessary

"""Model configurations"""
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--bias', type=bool, default=True)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--embed_dim', type=int, default=16, help='Embedding dimension of Pretrained model')
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--activation', type=str, default=None)

# LP head
parser.add_argument('--embed_dim_lp', type=int, default=32)

"""Training settings"""
parser.add_argument('--val_every', type=int, default=1)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--id', type=int, default=2)

"""Pretraining"""
parser.add_argument('--is_load', type=bool, default=False, help='Whether load model from checkpoints')
parser.add_argument('--pretrain_level', type=str, default="node", help='pretraining task level')
parser.add_argument('--pretrain_epochs', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)

# Zero-Shot Learning
parser.add_argument('--pretrained_model_path_ZSL', type=str, default="./pretrained_models")

# Node classification
parser.add_argument('--nc_epochs', type=int, default=120)
parser.add_argument('--lr_nc', type=float, default=0.001)
parser.add_argument('--weight_decay_nc', type=float, default=0.0000)
parser.add_argument('--nc_mode', type=str, default="transductive",
                    choices=["transductive", "inductive"])

# Link Prediction
parser.add_argument('--lp_epochs', type=int, default=200)
parser.add_argument('--lr_lp', type=float, default=0.001)
parser.add_argument('--weight_decay_lp', type=float, default=0.0)

# Graph classification
parser.add_argument('--gc_epochs', type=int, default=200)
parser.add_argument('--lr_gc', type=float, default=0.001)
parser.add_argument('--weight_decay_gc', type=float, default=0.0)

"""GPUs"""
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multiple gpus')

configs = parser.parse_args()

if not os.path.exists(configs.checkpoints):
    os.mkdir(configs.checkpoints)
if not os.path.exists(configs.log_dir):
    os.mkdir(configs.log_dir)
if configs.pretrained_model_path is None:
    configs.pretrained_model_path = f"Pretrain_{configs.pretrain_dataset}_model"
if configs.task_model_path is None:
    configs.task_model_path = f"{configs.task}_{configs.dataset}_model.pt"
if configs.log_name is None:
    configs.log_name = f"{configs.task}_{configs.dataset}.log"

print(configs)

if configs.task == 'Pretrain':
    pretrain_exp = Pretrain(configs)
    pretrain_exp.pretrain()
elif configs.task == 'NC':
    # exp = NodeClassification(configs, load=False, finetune=False)
    # exp.train()
    exp = ZeroShot(configs)
    exp.test()
elif configs.task == 'LP':
    exp = LinkPrediction(configs, load=True, finetune=True)
    exp.train()
else:
    raise NotImplementedError

torch.cuda.empty_cache()
