"""Targeted point perturbation attack."""

import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
import sys
sys.path.append('../')
sys.path.append('./')

from baselines.config import BEST_WEIGHTS
from baselines.config import MAX_PERTURB_BATCH as BATCH_SIZE
from baselines.dataset import ModelNet40Attack, ModelNetDataLoader, ScanObjectNN
from baselines.model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg, Pct, PointTransformerCls
from baselines.util.utils import str2bool, set_seed
from baselines.attack import CW_PFattack
from baselines.attack import CrossEntropyAdvLoss, UntargetedLogitsAdvLoss, AdvCrossEntropyLoss
from baselines.attack import L2Dist, ChamferDist, HausdorffDist
from baselines.attack import ClipPointsLinf, ClipLinf
import pickle


def attack():
    model.eval()
    all_ori_pc = []
    all_adv_pc = []
    all_real_lbl = []
    result = []
    num = 0
    for pc, normal, label, sharply in tqdm(test_loader):
        with torch.no_grad():
            pc, normal, label, sharply = pc.float().cuda(non_blocking=True), normal.float().cuda(non_blocking=True),\
                label.long().cuda(non_blocking=True), sharply.float().cuda(non_blocking=True)

        # attack!
        best_pc, success_num = attacker.attack(pc, None, normal, label)

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
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='baselines/official_data/modelnet40_normal_resampled')
    parser.add_argument('--model', type=str, default='pointnet', metavar='MODEL',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv', 'pointtransformer', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]. '
                             'If not specified, judge from data_root')
    parser.add_argument('--batch_size', type=int, default=1, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--budget', type=float, default=0.45,
                        help='clip budget')
    parser.add_argument('--sharply_root', type=str,
                        default='sharply_value/pointnet2/pointnet2_64players-concat.npz')
    #PT参数
    parser.add_argument('--model_name', default='Hengshuang', help='model name')
    parser.add_argument('--nneighbor', type=int, default=16)
    parser.add_argument('--nblocks', type=int, default=4)
    parser.add_argument('--transformer_dim', type=int, default=512)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--num_class', type=int, default=40)
    #
    #PCT参数
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')

    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40',
                                 'opt_mn40', 'conv_opt_mn40', 'aug_mn40', 'scanobject'])
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--adv_func', type=str, default='cross_entropy',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--drop_rate', type=float, default=0.01,
                        help='drop rate of points')
    parser.add_argument('--t', type=float, default=1.0,
                        choices=[1.0, 0.2])
    parser.add_argument('--kappa', type=float, default=30.,
                        help='min margin in logits adv loss')
    parser.add_argument('--attack_lr', type=float, default=1e-2,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step', type=int, default=1, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter', type=int, default=200, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--players', type=int, default=64, metavar='N',
                        help='num of players')
    parser.add_argument('--k_sharp', type=int, default=54, metavar='N',
                        help='num of k_sharp')
    parser.add_argument('--initial_const', type=float, default=10, help='')
    parser.add_argument('--pp', type=float, default=0.5, help='生成随机数为1的概率')
    parser.add_argument('--trans_weight', type=float, default=0.5, help='')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
        
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline')
    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_point]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_point]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)

    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    cudnn.benchmark = True

    # build model
    if args.model.lower() == 'pointtransformer':
        model = PointTransformerCls(args) 
    elif args.model.lower() == 'pct':
        model = Pct(args) # 模型输入[B,3,N]
    elif args.model.lower() == 'dgcnn':
        model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.model.lower() == 'pointnet':
        model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.model.lower() == 'pointnet2':
        model = PointNet2ClsSsg(num_classes=40)
    elif args.model.lower() == 'pointconv':
        model = PointConvDensityClsSsg(num_classes=40)
    else:
        print('Model not recognized')
        exit(-1)

    # if args.model.lower() == 'pointtransformer':
    #     model = nn.DataParallel(model).cuda()
    # load model weight
    state_dict = torch.load(
        BEST_WEIGHTS[args.model], map_location='cpu')
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))

    if args.model.lower() == 'pointtransformer':
        # state_dict = {'module.'+k: v for k, v in state_dict['model_state_dict'].items()}
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # eliminate 'module.' in keys
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)

    # distributed mode on multiple GPUs!
    # much faster than nn.DataParallel
    # model = DistributedDataParallel(
    #             model.cuda(), device_ids=[args.local_rank], find_unused_parameters=True)
    model = model.cuda()
    # setup attack settings
    if args.adv_func == 'logits':
        adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = AdvCrossEntropyLoss()
    dist_func = ChamferDist()
    clip_func = ClipLinf(budget=args.budget)
    # hyper-parameters from their official tensorflow code
    attacker = CW_PFattack(model_name=args.model.lower(), model=model,  adv_func=adv_func,
                        dist_func=dist_func,players=args.players, 
                        k_sharp=args.k_sharp, initial_const = args.initial_const,
                        pp = args.pp, trans_weight = args.trans_weight,
                        attack_lr=args.attack_lr, binary_step=args.binary_step,
                        num_iter=args.num_iter, clip_func=clip_func)

    # prepare data
    if args.dataset=='mn40':
        test_set = ModelNetDataLoader(root=args.data_root, args=args, split='test', process_data=args.process_data)
        test_sampler = DistributedSampler(test_set, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False, sampler=test_sampler)
    else:
        test_set = ScanObjectNN(partition='test', num_points=args.num_points)
        test_sampler = DistributedSampler(test_set, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False, sampler=test_sampler)

    # run attack
    ori_data, attacked_data, real_label, success_num, result = attack()

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)

    # save results
    save_path = './attack/results/{}_{}/PF_attack'.\
        format(args.dataset, args.num_point)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = 'UPF_attack-{}-{}-success_{:.4f}-rank_{}.npz'.\
        format(args.model, args.budget,
               success_rate, args.local_rank)
    np.savez(os.path.join(save_path, save_name),
            ori_pc=ori_data.astype(np.float32),
            test_pc=attacked_data.astype(np.float32),
            test_label=real_label.astype(np.uint8))