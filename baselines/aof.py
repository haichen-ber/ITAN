import os
import pdb
import time
from torch._C import _llvm_enabled
from tqdm import tqdm
import copy
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from attack.util.dist_utils import ChamferDist

from dataset import ModelNetDataLoader, CustomModelNet40, ModelNet40Attack
from model import DGCNN, PointNetCls, feature_transform_reguliarzer, \
    PointNet2ClsSsg, PointConvDensityClsSsg
from torch.utils.data import DataLoader
from util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed

from config import BEST_WEIGHTS
from attack import CrossEntropyAdvLoss, LogitsAdvLoss, UntargetedLogitsAdvLoss
from attack import ClipPointsLinf, ChamferkNNDist


def pc_visualize(pc, file_name="visual/img.png"):
    data = pc.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')  
    ax.scatter(data[:,0], data[:,1], data[:, 2], marker='o')
    ax.view_init(elev=-90., azim=-90) 
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig(file_name)
    plt.show()
    plt.close()


def knn(x, k):
    """
    x:(B, 3, N)
    """
    #print(f"input shape:{x.shape}")
    with torch.no_grad():
        inner = -2 * torch.matmul(x.transpose(2, 1), x)  #(B, N, N)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)   #(B, 1, N)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)   #(B, N, N)

        vec = x.transpose(2, 1).unsqueeze(2) - x.transpose(2, 1).unsqueeze(1)
        dist = -torch.sum(vec**2, dim=-1)
        #print("distance check:", torch.allclose(pairwise_distance, dist))

        #print(f"dist shape:{pairwise_distance.shape}")
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_Laplace_from_pc(ori_pc):
    """
    ori_pc:(B, 3, N)
    """
    #print("shape of ori pc:",ori_pc.shape)
    pc = ori_pc.detach().clone()
    with torch.no_grad():
        pc = pc.to('cpu').to(torch.double)
        idx = knn(pc, 30)
        pc = pc.transpose(2, 1).contiguous()  #(B, N, 3)
        point_mat = pc.unsqueeze(2) - pc.unsqueeze(1)  #(B, N, N, 3)
        A = torch.exp(-torch.sum(point_mat.square(), dim=3))  #(B, N, N)
        mask = torch.zeros_like(A)
        mask.scatter_(2, idx, 1)
        mask = mask + mask.transpose(2, 1)
        mask[mask>1] = 1
        
        A = A * mask
        D = torch.diag_embed(torch.sum(A, dim=2))
        L = D - A
        e, v = torch.symeig(L, eigenvectors=True)
    return e.to(ori_pc), v.to(ori_pc)


def normalize_points(points):
    """points: [K, 3]"""
    points = points - torch.mean(points, 0, keepdim=True)  # center
    dist = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
    print(dist)
    points = points / dist  # scale

    return points


def need_clip(pc, ori_pc, budget=0.1):
    with torch.no_grad():
        diff = pc - ori_pc  # [B, 3, K]
        norm = torch.sum(diff ** 2, dim=1) ** 0.5  # [B, K]
        scale_factor = budget / (norm + 1e-9)  # [B, K]
        bt = scale_factor < 1.0
        bt = torch.sum(bt, dim=-1)
        mask = (bt > 0).to(torch.float)

    return mask


def attack():
    iter_num = 0
    at_num, total_num, trans_num = 0.0, 0.0, 0.0
    all_adv_pc = []
    all_real_lbl = []
    st = time.time()
    for data, label in tqdm(test_loader):
        iter_num += 1
        with torch.no_grad():
            data, label = data.transpose(2, 1).float().cuda(non_blocking=True), \
                label.long().cuda(non_blocking=True)
        ori_data = data.detach().clone()
        ori_data.requires_grad_(False)

        B = data.shape[0]
        K = data.shape[2]
        # record best results in binary search
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, 3, K))
        label_val = label.detach().cpu().numpy()  # [B]

        # perform binary search
        for binary_step in range(args.step):
            data = ori_data.clone().detach() + \
                torch.randn((B, 3, K)).cuda() * 1e-7
            Evs, V = get_Laplace_from_pc(data)
            projs = torch.bmm(data, V)   #(B, 3, N)
            hfc = torch.bmm(projs[..., args.low_pass:],V[..., args.low_pass:].transpose(2, 1)) #(B, 3, N)  
            lfc = torch.bmm(projs[..., :args.low_pass],V[..., :args.low_pass].transpose(2, 1))
            lfc = lfc.detach().clone()
            hfc = hfc.detach().clone()
            lfc.requires_grad_()
            hfc.requires_grad_(False)
            ori_lfc = lfc.detach().clone()
            ori_lfc.requires_grad_(False)
            ori_hfc = hfc.detach().clone()
            ori_hfc.requires_grad_(False)
            opt = optim.Adam([lfc], lr=args.lr,
                        weight_decay=0)
            # opt = optim.Adam([{'params': hfc, 'lr': 1e-2},
            #                     {'params': lfc, 'lr': 1e-2}], lr=args.lr,
            #              weight_decay=0)
            #attack training
            for i in range(args.epochs):
                adv_pc = lfc + hfc
                if args.model.lower() == 'pointnet':
                    logits, _, _ = model(adv_pc)
                    lfc_logits, _, _ = model(lfc)
                else:
                    logits = model(adv_pc)
                    lfc_logits = model(lfc)

                # record values!
                pred = torch.argmax(logits, dim=1)  # [B]
                lfc_pred = torch.argmax(lfc_logits, dim=1)  # [B]
                dist_val = torch.amax(torch.abs(
                    (adv_pc - data)), dim=(1, 2)).\
                    detach().cpu().numpy()  # [B]
                pred_val = pred.detach().cpu().numpy()  # [B]
                lfc_pred_val = lfc_pred.detach().cpu().numpy()  # [B]
                # print(pred_val, dist_val)


                input_val = adv_pc.detach().cpu().numpy()  # [B, 3, K]
                # update
                for e, (dist, pred, lfc_pred, label_e, ii) in \
                        enumerate(zip(dist_val, pred_val, lfc_pred_val, label_val, input_val)):
                    if pred != label_e and dist < o_bestdist[e] and lfc_pred != label_e:
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred
                        o_bestattack[e] = ii

                loss = 0.5 * adv_func(logits, label) + 0.5 * adv_func(lfc_logits, label)
                opt.zero_grad()
                loss.backward()
                opt.step()

                #clip 
                with torch.no_grad():
                    adv_pc = lfc + hfc
                    adv_pc.data = clip_func(adv_pc.detach().clone(), data)
                    coeff = torch.bmm(adv_pc, V)
                    hfc.data = torch.bmm(coeff[..., args.low_pass:],V[..., args.low_pass:].transpose(2, 1)) #(B, 3, N)  
                    lfc.data = torch.bmm(coeff[..., :args.low_pass],V[..., :args.low_pass].transpose(2, 1))

            torch.cuda.empty_cache()
            
        #adv_pc = clip_func(adv_pc, data)
        print("best linf distance:", o_bestdist)
        adv_pc = torch.tensor(o_bestattack).to(adv_pc)
        adv_pc = clip_func(adv_pc, data)

        if args.model.lower() == 'pointnet':
            logits, _, _ = model(adv_pc)
        else:
            logits = model(adv_pc)
        preds = torch.argmax(logits, dim=-1)

        if args.trans_model.lower() == 'pointnet':
            trans_logits, _, _ = trans_model(adv_pc)
        else:
            trans_logits = trans_model(adv_pc)
        trans_preds = torch.argmax(trans_logits, dim=-1)

        at_num += (preds != label).sum().item()
        trans_num += (trans_preds != label).sum().item()
        
        total_num += args.batch_size

        print(preds, "\n", trans_preds)
        if iter_num % 1 == 0:
            print(f"attack success rate:{at_num / total_num}, trans success rate: {trans_num / total_num}")

        best_pc = adv_pc.transpose(1, 2).contiguous().detach().cpu().numpy()
        all_adv_pc.append(best_pc)
        all_real_lbl.append(label.detach().cpu().numpy())

    et = time.time()
    print(f"attack success rate:{at_num / total_num}, trans success rate: {trans_num / total_num}, consuming time:{et-st} seconds")
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    # save results
    save_path = './attack/results/{}_{}/AOF/{}'.\
        format(args.dataset, args.num_point, args.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    success_rate = at_num / total_num
    save_name = 'AOF-{}-low_pass_{}-budget_{}-success_{:.4f}.npz'.\
        format(args.model, args.low_pass, args.budget,
               success_rate)
    np.savez(os.path.join(save_path, save_name),
             test_pc=all_adv_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.uint8))
    print(args)


# def aof():
#     iter_num = 0
#     at_num, total_num, trans_num = 0.0, 0.0, 0.0
#     all_adv_pc = []
#     all_real_lbl = []
#     st = time.time()
#     for data, label, _ in tqdm(test_loader):
#         iter_num += 1
#         with torch.no_grad():
#             data, label = data.transpose(2, 1).float().cuda(non_blocking=True), \
#                 label.long().cuda(non_blocking=True)

#         B = data.shape[0]
#         K = data.shape[2]
#         # record best results in binary search
#         o_bestdist = np.array([1e10] * B)
#         o_bestscore = np.array([-1] * B)
#         o_bestattack = np.zeros((B, 3, K))
#         label_val = label.detach().cpu().numpy()  # [B]

#         Evs, V = get_Laplace_from_pc(data)
#         projs = torch.bmm(data, V)   #(B, 3, N)
#         opt_projs = projs[..., :args.low_pass].detach().cpu().numpy()
#         hfc = torch.bmm(projs[..., args.low_pass:],V[..., args.low_pass:].transpose(2, 1)) #(B, 3, N)  
#         lfc = torch.bmm(projs[..., :args.low_pass],V[..., :args.low_pass].transpose(2, 1))
#         ori_lfc = lfc.detach().clone()
#         inputs = torch.tensor(opt_projs, dtype=torch.float, device=data.device)
#         inputs.requires_grad_()
#         ori_weight = torch.tensor(opt_projs, dtype=torch.float, device=data.device)
#         ori_weight.requires_grad_(False)
#         opt = optim.Adam([inputs], lr=args.lr,
#                      weight_decay=0)
#         #attack training
#         for i in range(args.epochs):
#             lfc = torch.bmm(inputs, V[..., :args.low_pass].transpose(2, 1))
#             adv_pc = lfc + hfc
#             if args.model.lower() == 'pointnet':
#                 logits, _, _ = model(adv_pc)
#                 lfc_logits, _, _ = model(lfc)
#             else:
#                 logits = model(adv_pc)
#                 lfc_logits = model(lfc)

#             # record values!
#             pred = torch.argmax(logits, dim=1)  # [B]
#             lfc_pred = torch.argmax(lfc_logits, dim=1)  # [B]
#             dist_val = torch.amax(torch.abs(
#                 (adv_pc - data)), dim=(1, 2)).\
#                 detach().cpu().numpy()  # [B]
#             pred_val = pred.detach().cpu().numpy()  # [B]
#             lfc_pred_val = lfc_pred.detach().cpu().numpy()  # [B]
#             input_val = adv_pc.detach().cpu().numpy()  # [B, 3, K]

#             # update
#             for e, (dist, pred, lfc_pred, label_e, ii) in \
#                     enumerate(zip(dist_val, pred_val, lfc_pred_val, label_val, input_val)):
#                 if pred != label_e and lfc_pred != label_e and dist < o_bestdist[e]:
#                     o_bestdist[e] = dist
#                     o_bestscore[e] = pred
#                     o_bestattack[e] = ii
                
#             loss = adv_func(logits, label) + adv_func(lfc_logits, label)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#             #clip 
#             with torch.no_grad():
#                 adv_pc = torch.bmm(inputs, V[..., :args.low_pass].transpose(2, 1))# + hfc
#                 adv_pc.data = clip_func(adv_pc.detach().clone(), ori_lfc)

#                 inputs.data = torch.bmm(adv_pc, V[..., :args.low_pass])   #(B, 3, N)
            
#             # clip_mask = need_clip(adv_pc, data, args.budget)

#             # with torch.no_grad():
#             #     diff = inputs - ori_weight
#             #     norm = torch.norm(diff, dim=-1)  #(B, 3)
#             #     #print(f"------------------norm:{norm}-------------")
#             #     scale_factor = args.eig_budget / (norm + 1e-9)
#             #     scale_factor = torch.clamp(scale_factor, max=1.)  # [B, 3]
#             #     diff = diff * scale_factor[:, :, None] 
#             #     inputs.data = (ori_weight + diff).data * clip_mask.view(-1, 1, 1) + inputs.data*(1 - clip_mask).view(-1, 1, 1)

#         #adv_pc = adv_pc + hfc
#         adv_pc = torch.tensor(o_bestattack).to(data)
#         # adv_pc = clip_func(adv_pc, data)
#         #adv_pc = adv_pc + hfc
#         #adv_pc = lfc
       
#         if args.model.lower() == 'pointnet':
#             logits, _, _ = model(adv_pc)
#         else:
#             logits = model(adv_pc)
#         preds = torch.argmax(logits, dim=-1)

#         if args.trans_model.lower() == 'pointnet':
#             trans_logits, _, _ = trans_model(adv_pc)
#         else:
#             trans_logits = trans_model(adv_pc)
#         trans_preds = torch.argmax(trans_logits, dim=-1)

#         at_num = (preds != label).sum().item()
#         trans_num = (trans_preds != label).sum().item()
        
#         total_num = args.batch_size

#         print(label)
#         if iter_num % 1 == 0:
#             print(f"attack success rate:{at_num} / {total_num}, trans success rate: {trans_num} / {total_num}")

#         best_pc = adv_pc.transpose(1, 2).contiguous().detach().cpu().numpy()
#         all_adv_pc.append(best_pc)
#         all_real_lbl.append(label.detach().cpu().numpy())
#         # np.save(f'visual/vis_aof.npy', adv_pc.detach().cpu().numpy())
#         # break

#     et = time.time()
#     print(f"attack success rate:{at_num / total_num}, trans success rate: {trans_num / total_num}, consuming time:{et-st} seconds")
#     all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
#     all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]

#     # save results
#     save_path = './attack/results/{}_{}/AOF/{}'.\
#         format(args.dataset, args.low_pass, args.model)
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     save_name = 'adv_pc.npz'
#     np.savez(os.path.join(save_path, save_name),
#              test_pc=all_adv_pc.astype(np.float32),
#              test_label=all_real_lbl.astype(np.uint8))
#     print(args)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='official_data/modelnet40_normal_resampled')
    parser.add_argument('--attack_data_root', type=str,
                        default='data/attack_data.npz')
    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--trans_model', type=str, default='pointnet',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40',
                        choices=['mn40'])
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate for the optimizer')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--low_pass', type=int, default=100,
                        help='low_pass number')
    parser.add_argument('--budget', type=float, default=0.18,
                        help='FGM attack budget')
    parser.add_argument('--eig_budget', type=float, default=1.5,
                        help='FGM attack budget')
    parser.add_argument('--step', type=int, default=2, metavar='N',
                        help='Number of binary search step')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    args = parser.parse_args()
    set_seed(1)

    # enable cudnn benchmark
    cudnn.benchmark = True
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_point]

    # build victim model
    if args.model.lower() == 'dgcnn':
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
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(BEST_WEIGHTS[args.model]))
    model.eval()


    # build transfer model
    if args.trans_model.lower() == 'dgcnn':
        trans_model = DGCNN(args.emb_dims, args.k, output_channels=40)
    elif args.trans_model.lower() == 'pointnet':
        trans_model = PointNetCls(k=40, feature_transform=args.feature_transform)
    elif args.trans_model.lower() == 'pointnet2':
        trans_model = PointNet2ClsSsg(num_classes=40)
    elif args.trans_model.lower() == 'pointconv':
        trans_model = PointConvDensityClsSsg(num_classes=40)
    else:
        print('Model not recognized')
        exit(-1)
    trans_model = nn.DataParallel(trans_model).cuda()
    trans_model.load_state_dict(torch.load(BEST_WEIGHTS[args.trans_model]))
    trans_model.eval()


    # prepare dataset
    test_set = ModelNetDataLoader(root=args.data_root, args=args, split='test', process_data=args.process_data)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=False)

    # test_set = CustomModelNet40('custom_data', num_points=args.num_point, normalize=True)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size,
    #                          shuffle=False, num_workers=8,
    #                          pin_memory=True, drop_last=False)

    # test_set = ModelNet40Attack(args.attack_data_root, num_points=args.num_point,
    #                             normalize=True)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size,
    #                          shuffle=False, num_workers=4,
    #                          pin_memory=True, drop_last=False)


    clip_func = ClipPointsLinf(budget=args.budget)
    adv_func = UntargetedLogitsAdvLoss(kappa=30.)
    dist_func = ChamferDist()

    attack()