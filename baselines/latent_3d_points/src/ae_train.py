"""Targeted kNN attack."""

import os
import time
import copy
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch3d.loss import chamfer_distance
from tensorboardX import SummaryWriter

import sys
BASE_DIR = '/home/lbb/DLProjects/IF-Defense/baselines'
sys.path.append(BASE_DIR)
sys.path.append('../')
sys.path.append('./')
from dataset import ModelNet40, ModelNetDataLoader
from util.utils import AverageMeter, get_lr, set_seed

from encoders_decoders import AutoEncoder

def train(start_epoch):

    best_test_CD = 10.0
    best_cd_epoch = 0
    best_weight = copy.deepcopy(model.state_dict())
    # training begins
    for epoch in range(start_epoch, args.epochs + 1):
        step_count = 0
        loss_save = AverageMeter()
        model.train()

        # one epoch begins
        for data, label in train_loader:
            step_count += 1
            # print(label)
            # if step_count > 15:
            #     continue
            B, K = data.shape[:2]
            with torch.no_grad():
                data, label = data.float().cuda(), label.long().cuda()
                # to [B, 3, N] point cloud
                data = data.transpose(1, 2).contiguous()


            ori_data = data.detach().clone()
            # data = ori_data.clone().detach() + \
            #     torch.randn((B, 3, K)).cuda() * 1e-6
            # print(label)
            batch_size = data.size(0)
            opt.zero_grad()

            
            pc_reconstr = model(data)
            loss, _ = criterion(pc_reconstr.transpose(2, 1), ori_data.transpose(2, 1))
            loss.backward()
            opt.step()

            # statistics accumulation
            loss_save.update(loss.item(), batch_size)
            if epoch % 10 == 0 and step_count % args.print_iter == 0:
                print('Epoch {}, step {}, lr: {:.6f}\n'
                        'CD distance: {:.6f}'.
                        format(epoch, step_count, get_lr(opt),
                                loss_save.avg))
                torch.cuda.empty_cache()

        # eval
        if epoch % 50 == 0:
            CD_avg = test()
            if CD_avg < best_test_CD:
                best_test_CD = CD_avg
                best_cd_epoch = epoch
                best_weight = copy.deepcopy(model.state_dict())

            print('Epoch {}, chamfer distance {:.4f}\nCurrent best chamfer distance {:.4f} at epoch {}'.
                  format(epoch, CD_avg, best_test_CD, best_cd_epoch))
            torch.cuda.empty_cache() 
        scheduler.step(epoch)

        if epoch % 500 == 0:
            # save the best model
            torch.save(best_weight,
                    os.path.join(logs_dir,
                                    'BEST_model{}_CD_{:.4f}.pth'.
                                    format(best_cd_epoch, best_test_CD)))

def test():
    model.eval()
    CD_save = AverageMeter()
    with torch.no_grad():
        i = 0
        for data, label in test_loader:
            i += 1
            # print(label)
            # if i > 1:
            #     continue
            data, label = data.float().cuda(), label.long().cuda()
            # to [B, 3, N] point cloud
            data = data.transpose(1, 2).contiguous()
            batch_size = data.size(0)
                
            data_reconstr = model(data)
            if i in [1, 10, 20, 25]:
                np.save(f'vis_ae{i}.npy', data_reconstr.detach().cpu().numpy())
            CD, _ = criterion(data_reconstr.transpose(2, 1), data.transpose(2, 1))
            CD_save.update(CD.item(), batch_size)

    print('Test Avg Chamfer Distance: {:.4f}'.format(CD_save.avg))
    return CD_save.avg


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud AutoEncoder')
    parser.add_argument('--data_root', type=str,
                        default='official_data/modelnet40_normal_resampled')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40', 'opt_mn40', 'conv_opt_mn40'])
    parser.add_argument('--epochs', type=int, default=10001, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--batch_size', type=int, default=200, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='lr in train autoencoder')
    parser.add_argument('--num_iter', type=int, default=2500, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--print_iter', type=int, default=40,
                        help='Print interval')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    args = parser.parse_args()
    set_seed(1)
    print(args)

    cudnn.benchmark = True

    # build model
    model = AutoEncoder(k=3)

    
    model = torch.nn.DataParallel(model).cuda()

    # use Adam optimizer, cosine lr decay
    opt = optim.Adam(model.parameters(), lr=args.lr,
                     weight_decay=1e-6)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    # prepare data
    args.data_root = os.path.join(BASE_DIR, args.data_root)
    train_set = ModelNetDataLoader(root=args.data_root, args=args, split='train', process_data=args.process_data)
    test_set = ModelNetDataLoader(root=args.data_root, args=args, split='test', process_data=args.process_data)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=10, drop_last=False)

    # train_set = ModelNet40(args.data_root, num_points=args.num_points,
    #                        normalize=True, partition='train')
    # train_loader = DataLoader(train_set, batch_size=args.batch_size,
    #                           shuffle=True, num_workers=8,
    #                           pin_memory=True, drop_last=True)

    # test_set = ModelNet40(args.data_root, num_points=args.num_points,
    #                       normalize=True, partition='test')
    # test_loader = DataLoader(test_set, batch_size=args.batch_size,
    #                          shuffle=False, num_workers=8,
    #                          pin_memory=True, drop_last=False)

    # loss function using cross entropy without label smoothing
    criterion = chamfer_distance
    args.model = 'AE'

    # save folder
    start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logs_dir = "logs/{}/{}/{}_{}".format(args.dataset, args.model,
                                         start_datetime, args.num_points)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    logger = SummaryWriter(os.path.join(logs_dir, 'logs'))

    # start training
    train(1)
