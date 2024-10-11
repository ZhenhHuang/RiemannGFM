#!bin/bash

python main.py \
--task "Few-NC" \
--k_shot 5 \
--shot_epochs 25 \
--lr_few_nc 0.01 \
--query_set "GitHub" \
--pretrain_dataset 'ogbn-arxiv' 'computers' 'Physics' \
--pretrained_model_path "Pretrain_ogbn-arxiv_computers_Physics_model" \
--trained_model_path_FSL "5_shot_GitHub_trained_models" \
--root_path "datasets" \
--num_neighbors 20 10 \
--batch_size 64 \
--capacity 1000 \
--n_layers 2 \
--bias true \
--dropout 0.1 \
--embed_dim 32 \
--hidden_dim 256 \
--val_every 1 \
--patience 3 \
--is_load false