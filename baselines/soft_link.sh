#!/bin/bash

#custom pretrained model for resampled modelnet40 without augmentation
mkdir custom_pretrain
cd custom_pretrain
ln -s /home/ssd/big_data/lbb/mn40 ./mn40

#custom pretrained model for resampled modelnet40 with augmentation
cd ..
ln -s  /home/ssd/big_data/lbb/custom_pretrain_aug ./custom_pretrain_aug

#official normal resampled modelnet40
mkdir official_data
cd official_data
ln -s /home/ssd/big_data/lbb/modelnet40_normal_resampled ./modelnet40_normal_resampled

#link IFDefense pretrained model and data
cd ..
ln -s /home/ssd/big_data/IFDefense/pretrain ./pretrain
ln -s /home/ssd/big_data/IFDefense/data ./data