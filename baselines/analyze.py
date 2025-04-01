import os
import pdb
from tqdm import tqdm
import time
import copy
import argparse

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
#from tensorboardX import SummaryWriter

from dataset import ModelNetDataLoader
from model import DGCNN, PointNetCls, feature_transform_reguliarzer, \
    PointNet2ClsSsg, PointConvDensityClsSsg
from torch.utils.data import DataLoader
from util.utils import cal_loss, AverageMeter, get_lr, str2bool, set_seed
from attack import ChamferkNNDist

from config import BEST_WEIGHTS
from attack import CrossEntropyAdvLoss, LogitsAdvLoss, UntargetedLogitsAdvLoss
from attack import ClipPointsLinf


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


class FGM:
    """Class for FGM attack.
    """

    def __init__(self, model, adv_func, budget,
                 dist_metric='l2'):
        """FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            budget (float): \epsilon ball for FGM attack
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """

        self.model = model.cuda()
        self.model.eval()

        self.adv_func = adv_func
        self.budget = budget
        self.dist_metric = dist_metric.lower()

    def get_norm(self, x):
        """Calculate the norm of a given data x.

        Args:
            x (torch.FloatTensor): [B, 3, K]
        """
        # use global l2 norm here!
        norm = torch.sum(x ** 2, dim=[1, 2]) ** 0.5
        return norm

    def get_gradient(self, data_lfc, data_hfc, target, normalize=True):
        """Generate one step gradient.

        Args:
            data (torch.FloatTensor): batch pc, [B, 3, K]
            target (torch.LongTensor): target label, [B]
            normalize (bool, optional): whether l2 normalize grad. Defaults to True.
        """
        data_lfc = data_lfc.float().cuda()
        data_lfc.requires_grad_(True)
        data_hfc = data_hfc.float().cuda()
        data_hfc.requires_grad_(False)
        target = target.long().cuda()
        data = data_lfc + data_hfc

        # forward pass
        logits = self.model(data)
        if isinstance(logits, tuple):
            logits = logits[0]  # [B, class]
        pred = torch.argmax(logits, dim=-1)  # [B]

        # backward pass
        loss = self.adv_func(logits, target).mean()
        loss.backward()
        with torch.no_grad():
            grad = data_lfc.grad.detach()  # [B, 3, K]
            if normalize:
                norm = self.get_norm(grad)
                grad = grad / (norm[:, None, None] + 1e-9)
        return grad, pred

    def attack(self, data, target):
        pass

def clip(inputs, ori_weight):

    with torch.no_grad():
        diff = inputs - ori_weight
        norm = torch.norm(diff, dim=-1)  #(B, 3)
        scale_factor = 3.5 / (norm + 1e-9)
        scale_factor = torch.clamp(scale_factor, max=1.)  # [B, 3]
        diff = diff * scale_factor[:, :, None] 

    return ori_weight + diff

class IFGM(FGM):
    """Class for I-FGM attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter,
                 dist_metric='l2'):
        """Iterative FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(IFGM, self).__init__(model, adv_func, budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter

    def attack(self, data, target):
        """Iterative FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        pc = data.clone().detach().transpose(1, 2).contiguous()
        ori_pc = pc.detach().clone()

        low_pass, high_pass = -1, 1024
        _, V = get_Laplace_from_pc(pc)
        projs = torch.bmm(pc, V)   #(B, 3, N)
        if low_pass == -1:
            pc_lfc = torch.bmm(projs[..., :high_pass],V[..., :high_pass].transpose(2, 1)) #(B, 3, N) 
        else:
            pc_lfc = torch.bmm(projs[..., low_pass:high_pass],V[..., low_pass:high_pass].transpose(2, 1)) #(B, 3, N)  
        pc_hfc = pc - pc_lfc #(B, 3, N)  
        pc_lfc = pc_lfc + torch.randn((B, 3, K)).cuda() * 1e-7
        ori_pc_lfc = pc_lfc.detach().clone()
        target = target.long().cuda()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            #normalized_grad, pred = self.get_gradient(opt_projs, V_lfc, pc_hfc, target)
            normalized_grad, pred = self.get_gradient(pc_lfc, pc_hfc, target)
            success_num = (pred != target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                print('iter {}/{}, success: {}/{}'.
                      format(iteration, self.num_iter,
                             success_num, B))
                torch.cuda.empty_cache()
            perturbation = self.step_size * normalized_grad

            # add perturbation and clip
            with torch.no_grad():
                # opt_projs.data = opt_projs - perturbation
                # opt_projs.data = clip(opt_projs, ori_opt_projs)
                # pc_lfc = torch.bmm(opt_projs, V_lfc.transpose(2, 1))
                pc_lfc.data = pc_lfc - perturbation
                pc_lfc.data = self.clip_func(pc_lfc.detach().clone(), ori_pc_lfc)

        pc = pc_lfc + pc_hfc
        pc.data = self.clip_func(pc.detach().clone(), ori_pc)
        # end of iteration
        with torch.no_grad():
            logits = self.model(pc)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred != target).sum().item()
        print('Final success: {}/{}'.format(success_num, B))
        return pc.transpose(1, 2).contiguous().detach().cpu().numpy(), \
            success_num


class MIFGM(FGM):
    """Class for MI-FGM attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter, mu=1.,
                 dist_metric='l2'):
        """Momentum enhanced iterative FGM attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            mu (float): momentum factor
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(MIFGM, self).__init__(model, adv_func,
                                    budget, dist_metric)
        self.clip_func = clip_func
        self.step_size = step_size
        self.num_iter = num_iter
        self.mu = mu

    def attack(self, data, target):
        """Momentum enhanced iterative FGM attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        pc = data.clone().detach().transpose(1, 2).contiguous()
        ori_pc = pc.clone().detach()
        _, V = get_Laplace_from_pc(ori_pc)
        projs = torch.bmm(data, V)   #(B, 3, N)
        pc_lfc = torch.bmm(projs[..., :100],V[..., :100].transpose(2, 1)) #(B, 3, N)  
        pc_hfc = torch.bmm(projs[..., 100:],V[..., 100:].transpose(2, 1)) #(B, 3, N)  
        pc_lfc = pc_lfc + torch.randn((B, 3, K)).cuda() * 1e-7
        ori_pc_lfc = pc_lfc.detach().clone()
        target = target.long().cuda()
        momentum_g = torch.tensor(0.).cuda()

        # start iteration
        for iteration in range(self.num_iter):
            # gradient
            grad, pred = self.get_gradient(pc_lfc, pc_hfc, target, normalize=False)
            success_num = (pred != target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                print('iter {}/{}, success: {}/{}'.
                      format(iteration, self.num_iter,
                             success_num, B))
                torch.cuda.empty_cache()

            # grad is [B, 3, K]
            # normalized by l1 norm
            grad_l1_norm = torch.sum(torch.abs(grad), dim=[1, 2])  # [B]
            normalized_grad = grad / (grad_l1_norm[:, None, None] + 1e-9)
            momentum_g = self.mu * momentum_g + normalized_grad
            g_norm = self.get_norm(momentum_g)
            normalized_g = momentum_g / (g_norm[:, None, None] + 1e-9)
            perturbation = self.step_size * normalized_g

            # add perturbation and clip
            with torch.no_grad():
                pc_lfc = pc_lfc - perturbation
                pc_lfc = self.clip_func(pc_lfc, ori_pc_lfc)

        pc = pc_lfc + pc_hfc
        # end of iteration
        with torch.no_grad():
            logits = self.model(pc)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)
            success_num = (pred != target).sum().item()
        print('Final success: {}/{}'.format(success_num, B))
        return pc.transpose(1, 2).contiguous().detach().cpu().numpy(), \
            success_num


class PGD(IFGM):
    """Class for PGD attack.
    """

    def __init__(self, model, adv_func, clip_func,
                 budget, step_size, num_iter,
                 dist_metric='l2'):
        """PGD attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            clip_func (function): clipping method
            budget (float): \epsilon ball for IFGM attack
            step_size (float): attack step length
            num_iter (int): number of iteration
            dist_metric (str, optional): type of constraint. Defaults to 'l2'.
        """
        super(PGD, self).__init__(model, adv_func, clip_func,
                                  budget, step_size, num_iter,
                                  dist_metric)

    def attack(self, data, target):
        """PGD attack.

        Args:
            data (torch.FloatTensor): batch pc, [B, K, 3]
            target (torch.LongTensor): target label
        """
        # the only difference between IFGM and PGD is
        # the initialization of noise
        epsilon = self.budget / \
            ((data.shape[1] * data.shape[2]) ** 0.5)
        init_perturbation = \
            torch.empty_like(data).uniform_(-epsilon, epsilon)
        with torch.no_grad():
            init_data = data + init_perturbation
        return super(PGD, self).attack(init_data, target)


def attack():
    model.eval()
    trans_model.eval()
    all_adv_pc = []
    all_real_lbl = []
    num = 0
    at_num, trans_num = 0, 0
    total_num = 0
    for pc, label in tqdm(test_loader):
        with torch.no_grad():
            pc, label = pc.float().cuda(non_blocking=True), \
                label.long().cuda(non_blocking=True)

        # attack!
        best_pc, success_num = attacker.attack(pc, label)

        attacked_pc = torch.tensor(best_pc).float().cuda(non_blocking=True)
        attacked_pc = attacked_pc.transpose(2, 1).contiguous()
        if args.model.lower() == 'pointnet':
                logits, _, _ = model(attacked_pc)
        else:
            logits = model(attacked_pc)
        preds = torch.argmax(logits, dim=-1)
        print(preds, label)

        if args.trans_model.lower() == 'pointnet':
            trans_logits, _, _ = trans_model(attacked_pc)
        else:
            trans_logits = trans_model(attacked_pc)
        trans_preds = torch.argmax(trans_logits, dim=-1)
        # np.save(f'visual/vis_pgd.npy', best_pc.transpose(0, 2, 1))
        # print(trans_preds)
        # break

        total_num += args.batch_size 
        at_num += (preds != label).sum().item()
        trans_num += (trans_preds != label).sum().item()
        
        print(f"attack success rate:{at_num / total_num}, trans success rate: {trans_num / total_num}")

        # results
        num += success_num
        all_adv_pc.append(best_pc)
        all_real_lbl.append(label.detach().cpu().numpy())


    # accumulate results
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    return all_adv_pc, all_real_lbl, num


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='official_data/modelnet40_normal_resampled')
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
    parser.add_argument('--batch_size', type=int, default=32,
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
    parser.add_argument('--budget', type=float, default=0.1,
                        help='FGM attack budget')

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
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=False)


    clip_func = ClipPointsLinf(budget=args.budget)
    adv_func = UntargetedLogitsAdvLoss(kappa=15.)
    dist_func = ChamferkNNDist(chamfer_method='adv2ori',
                               knn_k=5, knn_alpha=1.05,
                               chamfer_weight=5., knn_weight=3.)

    args.step_size = args.budget * np.sqrt(args.num_point * 3)  / float(args.epochs)
    args.step_size = 0.01
    attacker = PGD(model, adv_func=adv_func,
                       clip_func=clip_func, budget=args.budget, step_size=args.step_size,
                       num_iter=args.epochs, dist_metric='l2')

    # run attack
    attacked_data, real_label, success_num = attack()

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)

    # save results
    save_path = './attack/results/{}_{}/PGDF/{}'.\
        format(args.dataset, args.num_point, args.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.adv_func = 'logits_kappa={}'.format(15.)
    save_name = 'PGDF-budget_{}-iter_{}'\
        '-success_{:.4f}.npz'.\
        format(args.budget, args.epochs, success_rate)
    np.savez(os.path.join(save_path, save_name),
             test_pc=attacked_data.astype(np.float32),
             test_label=real_label.astype(np.uint8))