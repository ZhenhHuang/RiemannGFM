#!bin/bash

python main.py \
--task "LP" \
--dataset "Cora" \
--root_path "./datasets" \
--pretrained_model_path "Pretrain_ogbn-arxiv_computers_Physics_model" \
--num_neighbors 30 30 \
--batch_size 64 \
--capacity 300 \
--n_layers 2 \
--bias true \
--dropout 0.1 \
--embed_dim 32 \
--hidden_dim 256 \
--val_every 1 \
--patience 5 \
--lr_lp 0.001 \
--embed_dim_lp 128