#!/bin/bash

low_passes=(30 40 60 70 80 90 110 120 130 140 150)
gpuid=$1
victim_model=$2

for i in ${!low_passes[@]}; do
    CUDA_VISIBLE_DEVICES=$gpuid python aof.py --model=$victim_model --budget=0.18 --low_pass=${low_passes[$i]} --batch_size=32
done
