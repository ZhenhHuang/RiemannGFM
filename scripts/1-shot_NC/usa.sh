#!bin/bash

python main.py \
--task "Few-NC" \
--k_shot 1 \
--shot_epochs 100 \
--lr_few_nc 0.01 \
--query_set "USA" \
--pretrain_dataset 'ogbn-arxiv' 'computers' 'Physics' \
--pretrained_model_path "Pretrain_ogbn-arxiv_computers_Physics_model" \
--trained_model_path_FSL "1_shot_USA_trained_models" \
--root_path "datasets" \
--num_neighbors 20 10 \
--batch_size 64 \
--capacity 1000 \
--n_layers 2 \
--bias true \
--dropout 0.0 \
--embed_dim 32 \
--hidden_dim 256 \
--task_hidden_dim 16 \
--val_every 1 \
--patience 3 \
--is_load false \
--exp_iters 5 \
 --drop_edge 0.0 \
 --drop_feats 0.0