import os
import pdb
import time
from tqdm import tqdm
import copy
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR


import sys
sys.path.append('../')
sys.path.append('./')
import pickle
from baselines.attack.util.dist_utils import ChamferDist
from baselines.dataset import ModelNetDataLoader, CustomModelNet40, ModelNet40Attack, ScanObjectNN
from baselines.model import DGCNN, PointNetCls, feature_transform_reguliarzer, \
    PointNet2ClsSsg, PointConvDensityClsSsg, Pct
from baselines.util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed

from baselines.config import BEST_WEIGHTS
from baselines.config import MAX_AOF_BATCH as BATCH_SIZE
from baselines.attack import SSCWAOF
from baselines.attack import CrossEntropyAdvLoss, LogitsAdvLoss, UntargetedLogitsAdvLoss
from baselines.attack import ClipPointsLinf, ChamferkNNDist, L2Dist

def normalize_points(points):
    """points: [K, 3]"""
    points = points - torch.mean(points, 0, keepdim=True)  # center
    dist = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
    print(dist)
    points = points / dist  # scale

    return points

def attack():
    model.eval()
    all_ori_pc = []
    all_adv_pc = []
    result = []
    all_real_lbl = []
    num = 0
    for pc, normal, label, sharply in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(non_blocking=True), \
                label.long().cuda(non_blocking=True)

        # attack!
        _, best_pc, success_num = attacker.attack(pc, None, label)
        # np.save(f'visual/vis_aof.npy', best_pc.transpose(0, 2, 1))

        # results
        num += success_num
        all_ori_pc.append(pc.detach().cpu().numpy())
        all_adv_pc.append(best_pc.detach().cpu().numpy())
        all_real_lbl.append(label.detach().cpu().numpy())

    # accumulate results
    all_ori_pc = np.concatenate(all_ori_pc, axis=0)  # [num_data, K, 3]
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    result.append(all_ori_pc)
    result.append(all_adv_pc)
    result.append(all_real_lbl)
    return all_ori_pc, all_adv_pc, all_real_lbl, num, result

if __name__ == "__main__":
    # Training settings
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='baselines/official_data/modelnet40_normal_resampled')
    parser.add_argument('--sharply_root', type=str,
                        default='sharply_value/pointnet2/pointnet2_64players-concat.npz')
    parser.add_argument('--attack_data_root', type=str,
                        default='data/attack_data.npz')
    parser.add_argument('--model', type=str, default='pct',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40',
                        choices=['mn40', 'aug_mn40', 'scanobject'])
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS',
                        help='Size of batch')
    
    #PCT参数
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')
    
    parser.add_argument('--num_iter', type=int, default=200, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--attack_lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate for the optimizer')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--low_pass', type=int, default=100,
                        help='low_pass number')
    parser.add_argument('--budget', type=float, default=0.18,
                        help='FGM attack budget')
    parser.add_argument('--GAMMA', type=float, default=0.25,
                        help='hyperparameter gamma')
    parser.add_argument('--binary_step', type=int, default=2, metavar='N',
                        help='Number of binary search step')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_point]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_point]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)

    # enable cudnn benchmark
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    cudnn.benchmark = True

    # build victim model
    if args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.model.lower() == 'pct':
        model = Pct(args) # 模型输入[B,3,N]
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif args.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
    else:
        print('Model not recognized')
        exit(-1)
    # model = nn.DataParallel(model).cuda()
    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    # model = DistributedDataParallel(
    #     model.cuda(), device_ids=[args.local_rank])
    model = model.cuda()
    model.eval()


    # prepare dataset
    # prepare data
    if args.dataset=='mn40':
        test_set = ModelNetDataLoader(root=args.data_root, args=args, split='test', process_data=args.process_data)
        test_sampler = DistributedSampler(test_set, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=False, sampler=test_sampler)
    else:
        test_set = ScanObjectNN(partition='test', num_points=args.num_points)
        test_sampler = DistributedSampler(test_set, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=False, sampler=test_sampler)

    # test_set = ModelNet40Attack(args.attack_data_root, num_points=args.num_point,
    #                             normalize=True)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size,
    #                          shuffle=False, num_workers=4,
    #                          pin_memory=True, drop_last=False)


    clip_func = ClipPointsLinf(budget=args.budget)
    adv_func = UntargetedLogitsAdvLoss(kappa=30.)
    dist_func = L2Dist()
    attacker = SSCWAOF(model, adv_func, dist_func,
                         attack_lr=args.attack_lr,
                         binary_step=args.binary_step,
                         num_iter=args.num_iter, GAMMA=args.GAMMA,
                         low_pass = args.low_pass,
                         clip_func=clip_func)

    print(len(test_set))
    # run attack
    ori_data, attacked_data, real_label, success_num, result = attack()

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)

    # save results
    save_path = './attack/results/{}_{}/SSAOF'.\
        format(args.dataset, args.num_point)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = 'USSAOF-{}-{}-{}-GAMMA_{}-success_{:.4f}-rank_{}.npz'.\
        format(args.model, args.budget,args.low_pass, args.GAMMA,
               success_rate, args.local_rank)
    np.savez(os.path.join(save_path, save_name),
            ori_pc=ori_data.astype(np.float32),
            test_pc=attacked_data.astype(np.float32),
            test_label=real_label.astype(np.uint8))
