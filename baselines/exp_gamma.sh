#!/bin/bash

gammas=(0.75 1.0)
victim_models=("pointnet" "pointnet2" "pointconv" "dgcnn")

for i in ${!gammas[@]}; do
    for j in ${!victim_models[@]}; do
        NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29504 attack_scripts/untargeted_aof_attack.py --process_data --model=${victim_models[$j]} --GAMMA=${gammas[$i]}
    done
done