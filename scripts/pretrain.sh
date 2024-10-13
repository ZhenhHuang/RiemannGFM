#!bin/bash

python main.py \
--task "Pretrain" \
--pretrain_dataset "ogbn-arxiv" "computers" "Physics" \
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
--val_every 1 \
--patience 3 \
--is_load false \
--pretrain_iters 3 \
--pretrain_epochs 3 \
--lr 0.01 \
--weight_decay 0.0