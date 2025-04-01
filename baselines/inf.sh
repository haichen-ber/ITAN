#!/bin/bash

gpuid=$1
data=$2
prefix=$3
victim=$4
dataset=$5
victim_models=("pointnet" "pointnet2" "pointconv" "dgcnn")
ASR_sum=0

if [[ $dataset -eq "" ]]
then
    echo "The dataset parameter is None, use default mn40."
    dataset="mn40"
else
    echo $dataset
fi

for j in ${!victim_models[@]}; do
    asr=$(CUDA_VISIBLE_DEVICES=$gpuid python inference.py --model=${victim_models[$j]} --batch_size=64 --data_root=$data --prefix=$prefix --dataset=$dataset)
    echo $asr
    if [[ "${victim_models[$j]}" == "$victim" ]]
    then
        ASR=$asr
    else
        ASR_sum=`echo $ASR_sum + $asr | bc`
    fi
done

denom=3
echo $ASR_sum
TR=`echo "scale=3; $ASR_sum / $denom" | bc`
echo "Transferability:$TR"
echo "ASR for $victim:$ASR"