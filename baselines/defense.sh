#!/bin/bash

data_folder=$1
prefix=$2
victim=$3
gpuid=$4

data="${data_folder}/${prefix}-concat.npz"
# echo $data

defense_types=("srs" "dup" "sor")
for i in ${!defense_types[@]}; do
    tp=${defense_types[$i]}
    echo "$tp defense and inference"
    CUDA_VISIBLE_DEVICES=$gpuid python defend_npz.py --data_root=$data --defense=$tp
    tp_folder="$data_folder/$tp"
    tp_prefix="${tp}_${prefix}"
    # echo $tp_folder
    # echo $tp_prefix
    ./inf.sh $gpuid $tp_folder $tp_prefix $victim
done

echo "ConvOnet Defense and inference"
cd ../ConvONet
CUDA_VISIBLE_DEVICES=$gpuid python ../ConvONet/opt_defense.py --sample_npoint=1024 --train=False --rep_weight=500.0 --data_root=$data
tp=convonet_opt
tp_folder="$data_folder/ConvONet-Opt"
tp_prefix="${tp}-${prefix}"
echo $tp_prefix
echo $tp_folder
cd ../baselines
./inf.sh $gpuid $tp_folder $tp_prefix $victim

cd ../ONet
echo "Onet-Remesh Defense and inference"
CUDA_VISIBLE_DEVICES=$gpuid python ../ONet/remesh_defense.py --sample_npoint=1024 --train=False --data_root=$data
tp=onet_remesh
tp_folder="$data_folder/ONet-Mesh"
tp_prefix="${tp}-${prefix}"
cd ../baselines
./inf.sh $gpuid $tp_folder $tp_prefix $victim

cd ../ONet
echo "Onet-Opt Defense and inference"
CUDA_VISIBLE_DEVICES=$gpuid python ../ONet/opt_defense.py --sample_npoint=1024 --train=False --rep_weight=500.0 --data_root=$data
tp=onet_opt
tp_folder="$data_folder/ONet-Opt"
tp_prefix="${tp}-${prefix}"
cd ../baselines
./inf.sh $gpuid $tp_folder $tp_prefix $victim