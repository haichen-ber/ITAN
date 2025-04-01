#!/bin/bash

gpuid=$1
data=$2
victim_models=("pointnet")


for j in ${!victim_models[@]}; do
	CUDA_VISIBLE_DEVICES=$gpuid python inference.py --model=${victim_models[$j]} --data_root=$data
done
