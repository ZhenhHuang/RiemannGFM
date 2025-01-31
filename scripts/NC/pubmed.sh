#!bin/bash

python main.py \
--task "NC" \
--dataset "PubMed" \
--pretrain_dataset 'ogbn-arxiv' 'computers' 'Physics' \
--pretrained_model_path "Pretrain_ogbn-arxiv_computers_Physics_model" \
--root_path "datasets" \
--num_neighbors 20 10 \
--batch_size 32 \
--capacity 1 \
--n_layers 2 \
--bias true \
--dropout 0.1 \
--embed_dim 32 \
--hidden_dim 256 \
--nc_hidden_dim 32 \
--val_every 1 \
--patience 10 \
--is_load false \
--pretrain_epochs 3 \
--lr_nc 0.01 \
--weight_decay_nc 0.0 \
--nc_epochs 120 \
 --drop_edge 0.2 \
 --drop_feats 0.3