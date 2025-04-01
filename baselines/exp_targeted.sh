#!/bin/bash


NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29504 attack_scripts/targeted_aof_attack.py --model=dgcnn