import os
import time
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pytorch3d.loss import chamfer_distance

import sys
sys.path.append('../')
sys.path.append('./')

from config import BEST_WEIGHTS
from config import MAX_KNN_BATCH as BATCH_SIZE
from dataset import ModelNetDataLoader, ModelNet40Attack
from model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg
from util.utils import str2bool, set_seed
from attack import CrossEntropyAdvLoss, UntargetedLogitsAdvLoss
from attack import ChamferDist
from attack import ClipPointsLinf
from latent_3d_points.src import encoders_decoders

def attack():
    model.eval()
    ae_model.eval()
    all_adv_pc = []
    all_real_lbl = []
    num = 0
    at_num, total_num, trans_num = 0.0, 0.0, 0.0
    victims_list = [0, 5, 35, 2, 8, 33, 22, 37, 4, 30]
    trans_total = 0
    for data, label in tqdm(test_loader):
        with torch.no_grad():
            data, label = data.float().cuda(non_blocking=True), \
                label.long().cuda(non_blocking=True)

        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        data = data.transpose(1, 2).contiguous()
        ori_data = data.clone().detach()
        ori_data.requires_grad = False
        label_val = label.detach().cpu().numpy()  # [B]

        # record best results in binary search
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, 3, K))


        # perform binary search
        for binary_step in range(args.step):
            # init variables with small perturbation
            adv_data = ori_data.clone().detach() + \
                torch.randn((B, 3, K)).cuda() * 1e-7
            adv_data.requires_grad_()
            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)
            opt = optim.Adam([adv_data], lr=args.attack_lr, weight_decay=0.)

            adv_loss = torch.tensor(0.).cuda()
            dist_loss = torch.tensor(0.).cuda()

            total_time = 0.
            forward_time = 0.
            backward_time = 0.
            update_time = 0.

            # one step in binary search
            for iteration in range(args.num_iter):
                t1 = time.time()

                # forward passing
                logits = model(adv_data)  # [B, num_classes]
                adv_data_constr = ae_model(adv_data)
                # np.save(f'visual/vis_advpc.npy', adv_data_constr.detach().cpu().numpy())
                # return
                # with torch.no_grad():
                #     print("CD Distance:", dist_func(adv_data_constr.transpose(2, 1), adv_data.transpose(2, 1)))
                ae_logits = model(adv_data_constr)

                if isinstance(logits, tuple):  # PointNet
                    logits = logits[0]
                    ae_logits = ae_logits[0]

                t2 = time.time()
                forward_time += t2 - t1

                # print
                pred = torch.argmax(logits, dim=1)  # [B]
                ae_pred = torch.argmax(ae_logits, dim=1)  # [B]
                success_num = ((pred != label) * (ae_pred != label)).sum().item()
                #success_num = (ae_pred != label).sum().item()
                #print(success_num)
                if iteration % (args.num_iter // 5) == 0:
                    print('Step {}, iteration {}, success {}/{}\n'.
                          format(binary_step, iteration, success_num, B))
                    with torch.no_grad():
                        print("CD Distance:", dist_func(adv_data_constr.transpose(2, 1), adv_data.transpose(2, 1)))


                # record values!
                dist_val = torch.amax(torch.abs(
                    (adv_data - ori_data)), dim=(1, 2)).\
                    detach().cpu().numpy()  # [B]
                pred_val = pred.detach().cpu().numpy()  # [B]
                ae_pred_val = ae_pred.detach().cpu().numpy()  # [B]
                input_val = adv_data.detach().cpu().numpy()  # [B, 3, K]

                # update
                for e, (dist, pred, ae_pred, label_e, ii) in \
                        enumerate(zip(dist_val, pred_val, ae_pred_val, label_val, input_val)):
                    if pred != label_e and dist < bestdist[e] and ae_pred != label_e:
                        bestdist[e] = dist
                        bestscore[e] = pred
                    if pred != label_e and dist < o_bestdist[e] and ae_pred != label_e:
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred
                        o_bestattack[e] = ii

                t3 = time.time()
                update_time += t3 - t2

                opt.zero_grad()

                adv_loss = 0.75*adv_func(logits, label).mean() + 0.25*ae_adv_func(ae_logits, label).mean()
               
                loss = adv_loss
                loss.backward()
                opt.step()

                # clipping and projection!
                adv_data.data = clip_func(adv_data.clone().detach(),
                                               ori_data)

                t4 = time.time()
                backward_time += t4 - t3
                total_time += t4 - t1

                if iteration % 100 == 0:
                    print('total time: {:.2f}, for: {:.2f}, '
                          'back: {:.2f}, update: {:.2f}'.
                          format(total_time, forward_time,
                                 backward_time, update_time))
                    total_time = 0.
                    forward_time = 0.
                    backward_time = 0.
                    update_time = 0.
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()

        print("best linf distance:", o_bestdist)
        print(label)
        adv_pc = torch.tensor(o_bestattack).to(adv_data)
        adv_pc.data = clip_func(adv_pc.clone().detach(), data)
        #victim model inference
        if args.model.lower() == 'pointnet':
            best_logits, _, _ = model(adv_pc)
        else:
            best_logits = model(adv_pc)
        pred = torch.argmax(best_logits, dim=-1)  # [B]
        success_num = (pred != label).\
            sum().detach().cpu().item()

        #trans model inference
        if args.trans_model.lower() == 'pointnet':
            trans_logits, _, _ = trans_model(adv_pc)
        else:
            trans_logits = trans_model(adv_pc)
        trans_pred = torch.argmax(trans_logits, dim=-1)  # [B]
        trans_success_num = 0
        # for i in range(B):
        #     if label_val[i] in victims_list and trans_pred[i] != label[i]:
        #         trans_success_num += 1
        #     if label_val[i] in victims_list:
        #         trans_total += 1
        trans_success_num = (trans_pred != label).\
            sum().detach().cpu().item()

        at_num += success_num
        trans_num += trans_success_num
        total_num += B
        print(f"success:{success_num}/{data.shape[0]}, trans success:{trans_success_num}/{data.shape[0]},\
                 attack success rate:{at_num/total_num}, trans success rate:{trans_num / total_num}")

        best_pc = adv_pc.transpose(1, 2).contiguous().detach().cpu().numpy()
        # np.save(f'visual/vis_advpc.npy', adv_pc.detach().cpu().numpy())
        # return

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
                        default='official_data/modelnet40_normal_resampled/')
    parser.add_argument('--attack_data_root', type=str,
                        default='data/attack_data.npz')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--trans_model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40', 'opt_mn40', 'conv_opt_mn40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS',
                        help='Size of batch')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa', type=float, default=30.,
                        help='min margin in logits adv loss')
    parser.add_argument('--budget', type=float, default=0.18,
                        help='FGM attack budget')
    parser.add_argument('--attack_lr', type=float, default=1e-2,
                        help='lr in attack training optimization')
    parser.add_argument('--num_iter', type=int, default=200, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--step', type=int, default=2, metavar='N',
                        help='Number of binary search step')
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

    cudnn.benchmark = True

    # build model
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

    # load model weight
    state_dict = torch.load(BEST_WEIGHTS[args.model])
    print('Loading weight {}'.format(BEST_WEIGHTS[args.model]))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # eliminate 'module.' in keys
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model = torch.nn.DataParallel(model).cuda()

    # build transfer model
    # build model
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

    trans_model = torch.nn.DataParallel(trans_model).cuda()
    trans_model.load_state_dict(torch.load(BEST_WEIGHTS[args.trans_model]))
    trans_model.eval()


    #AutoEncoder model
    ae_model = encoders_decoders.AutoEncoder(3)
    #ae_path = "latent_3d_points/src/logs/mn40/AE/2022-01-02 09:36:21_1024/BEST_model250_CD_0.0027.pth"
    ae_path = "latent_3d_points/src/logs/mn40/AE/2021-12-31 15:15:52_1024/BEST_model9800_CD_0.0038.pth"
    ae_state_dict = torch.load(ae_path)
    print('Loading ae weight {}'.format(ae_path))
    try:
        ae_model.load_state_dict(ae_state_dict)
    except RuntimeError:
        ae_state_dict = {k[7:]: v for k, v in ae_state_dict.items()}
        ae_model.load_state_dict(ae_state_dict)

    ae_model = torch.nn.DataParallel(ae_model).cuda()


    # setup attack settings
    if args.adv_func == 'logits':
        adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)
    else:
        adv_func = CrossEntropyAdvLoss()
    ae_adv_func = UntargetedLogitsAdvLoss(kappa=30.)
    clip_func = ClipPointsLinf(budget=args.budget)
    dist_func = ChamferDist()

    # attack
    test_set = ModelNetDataLoader(root=args.data_root, args=args, split='test', process_data=args.process_data)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=False)
    # test_set = ModelNet40Attack(args.attack_data_root, num_points=args.num_point,
    #                             normalize=True)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size,
    #                          shuffle=False, num_workers=4,
    #                          pin_memory=True, drop_last=False)

    # run attack
    attacked_data, real_label, success_num = attack()

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)

    # save results
    save_path = './attack/results/{}_{}/AdvPC'.\
        format(args.dataset, args.num_point)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if args.adv_func == 'logits':
        args.adv_func = 'logits_kappa={}'.format(args.kappa)
    save_name = 'AdvPC-{}-{}-budget_{}_success_{:.4f}.npz'.\
        format(args.model, args.adv_func, args.budget,
               success_rate)
    np.savez(os.path.join(save_path, save_name),
             test_pc=attacked_data.astype(np.float32),
             test_label=real_label.astype(np.uint8))
