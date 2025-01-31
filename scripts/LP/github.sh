#!bin/bash

python main.py \
--task "LP" \
--dataset "GitHub" \
--root_path "./datasets" \
--pretrained_model_path "Pretrain_ogbn-arxiv_computers_Physics_model" \
--id 2 \
--num_neighbors 20 10 \
--batch_size 16 \
--capacity 100 \
--n_layers 2 \
--bias true \
--dropout 0.1 \
--embed_dim 32 \
--hidden_dim 256 \
--val_every 1 \
--patience 5 \
--lr_lp 0.01 \
--embed_dim_lp 64