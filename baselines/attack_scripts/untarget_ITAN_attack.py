"""Training file for the victim models"""
import os
import argparse
import sys
sys.path.append('../')
sys.path.append('./')
import torch
import pickle
import torch.nn as nn
from baselines.config import BEST_WEIGHTS
from baselines.config import MAX_PERTURB_BATCH as BATCH_SIZE
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from baselines.attack import ChamferDist,ChamferkNNDhusdorfist, HausdorffDist, ChamferL2
from baselines.dataset import ModelNet40, ModelNetDataLoader, ScanObjectNN
from baselines.model import DGCNN, PointNetCls, \
    PointNet2ClsSsg, PointConvDensityClsSsg, Pct
from torch.utils.data import DataLoader
from baselines.util.utils import str2bool, set_seed
from baselines.attack.CW import UAE_pretrain
import numpy as np
from baselines.attack import ClipLinf
from tqdm import tqdm
from baselines.latent_3d_points.AE_z import AutoEncoder_z
from baselines.model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg, Pct, PointTransformerCls
from baselines.util.utils import str2bool, set_seed
from baselines.attack import CWPerturb
from baselines.attack import CrossEntropyAdvLoss, UntargetedLogitsAdvLoss
from baselines.attack import L2Dist
from baselines.attack import ClipPointsLinf
from baselines.attack  import CWUKNN, CWAOF, SSCWAOF
from baselines.attack  import ChamferkNNDist
from baselines.attack  import ProjectInnerClipLinf
from baselines.latent_3d_points.src import encoders_decoders
from baselines.attack import CWUAdvPC
from baselines.attack import CW_PFattack
from baselines.attack import AdvCrossEntropyLoss
from baselines.attack import ChamferDist, HausdorffDist


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='baselines/official_data/modelnet40_normal_resampled')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--dataset', type=str, default='scanobject',
                        choices=['mn40','scanobject'])
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings in DGCNN')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use in DGCNN')
    parser.add_argument('--print_iter', type=int, default=50,
                        help='Print interval')
    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    
    ## pct
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')

    ## 3d-adv
    parser.add_argument('--adv_func', type=str, default='logits',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--kappa_3d_adv', type=float, default=30.,
                        help='min margin in logits adv loss')
    parser.add_argument('--attack_lr_3d_adv', type=float, default=1e-2,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step_3d_adv', type=int, default=10, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter_3d_adv', type=int, default=500, metavar='N',
                        help='Number of iterations in each search step')
    
    #### knn 
    parser.add_argument('--kappa_knn', type=float, default=15.,
                        help='min margin in logits adv loss')
    parser.add_argument('--attack_lr_knn', type=float, default=1e-3,
                        help='lr in CW optimization')
    parser.add_argument('--num_iter_knn', type=int, default=2500, metavar='N',
                        help='Number of iterations in each search step')
    
    ### advpc
    parser.add_argument('--ae_model_path', type=str,
                        default='baselines/latent_3d_points/src/logs/mn40/AE/BEST_model9800_CD_0.0038.pth')
    parser.add_argument('--kappa_advpc', type=float, default=30.,
                        help='min margin in logits adv loss')
    parser.add_argument('--GAMMA_advpc', type=float, default=0.25,
                        help='hyperparameter gamma')
    parser.add_argument('--binary_step_advpc', type=int, default=2, metavar='N',
                        help='Number of binary search step')
    parser.add_argument('--attack_lr_advpc', type=float, default=1e-2,
                        help='lr in attack training optimization')
    parser.add_argument('--num_iter_advpc', type=int, default=200, metavar='N',
                        help='Number of iterations in each search step')
    
    ### pf-attack
    parser.add_argument('--adv_func_pf', type=str, default='cross_entropy',
                        choices=['logits', 'cross_entropy'],
                        help='Adversarial loss function to use')
    parser.add_argument('--drop_rate', type=float, default=0.01,
                        help='drop rate of points')
    parser.add_argument('--t_pf', type=float, default=1.0,
                        choices=[1.0, 0.2])
    parser.add_argument('--players', type=int, default=64, metavar='N',
                        help='num of players')
    parser.add_argument('--k_sharp', type=int, default=54, metavar='N',
                        help='num of k_sharp')
    parser.add_argument('--attack_lr_pf', type=float, default=1e-2,
                        help='lr in CW optimization')
    parser.add_argument('--binary_step_pf', type=int, default=1, metavar='N',
                        help='Binary search step')
    parser.add_argument('--num_iter_pf', type=int, default=200, metavar='N',
                        help='Number of iterations in each search step')
    parser.add_argument('--initial_const_pf', type=float, default=10, help='')
    parser.add_argument('--pp_pf', type=float, default=0.5, help='生成随机数为1的概率')
    parser.add_argument('--trans_weight_pf', type=float, default=0.5, help='')

    ### aof
    parser.add_argument('--attack_lr_aof', type=float, default=1e-2, metavar='LR',
                        help='learning rate for the optimizer')
    parser.add_argument('--low_pass_aof', type=int, default=100,
                        help='low_pass number')
    parser.add_argument('--GAMMA_aof', type=float, default=0.25,
                        help='hyperparameter gamma')
    parser.add_argument('--binary_step_aof', type=int, default=2, metavar='N',
                        help='Number of binary search step')
    parser.add_argument('--num_iter_aof', type=int, default=200, metavar='N',
                        help='number of epochs to train ')

    ####### parameters of generator ########
    parser.add_argument('--finetune_mode', type=str, default='noise',
                        choices=['noise', 'G'])
    parser.add_argument('--attack_methods', type=str, default='aof',
                        choices=['3d-adv', 'knn', 'advpc', 'pf-attack', 'aof', 'ss-aof'])
    parser.add_argument('--init_noise', default=True, help='use normals')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=4,
                        help='Size of batch')
    parser.add_argument('--epoch', type=int, default=200, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--epoch_test', type=int, default=50, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--binary_step', type=int, default=4, metavar='N',
                        help='number of epochs to train ')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate for the optimizer')
    parser.add_argument('--model', type=str, default='dgcnn',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv','pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]')
    parser.add_argument('--div_loss', default=True, type=bool)
    parser.add_argument('--trans_loss', default=True, type=bool)
    parser.add_argument('--lam_div', default=5., type=float, help='weight for div_loss')
    parser.add_argument('--lam_trans', default=5., type=float, help='weight for trans_loss')
    parser.add_argument('--lam_dis', default=10., type=float, help='weight for chamfer_dis')
    parser.add_argument('--z_dim', default=3, type=int, help='dimension of the latent vector')
    parser.add_argument('--epsilon', default=0.18, type=float, help='perturbation constraint')
    parser.add_argument('--num_category', default=15, type=int, choices=[10, 40,15],  help='training on ModelNet10/40/scanobject')
    parser.add_argument('--target_layer', type=str, default='dp2',
                    help='target layer : '
                         'dropout for pointnet,'
                         'drop2 for pointnet2,'
                         'dp2 for dgcnn,'
                         'drop2 for pointconv'
                         'dp2 for pct')
    parser.add_argument('--save_dir', default='mg_weight/scanobject', type=str, help='directory for saving model weights')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--sharply_root', type=str,
                        default='sharply_value/pointnet/pointnet_25players-concat.npz')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline')
    args = parser.parse_args()
    BATCH_SIZE = BATCH_SIZE[args.num_point]
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][args.num_point]
    if args.batch_size == -1:
        args.batch_size = BATCH_SIZE[args.model]
    set_seed(1)
    print(args)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    # enable cudnn benchmark
    cudnn.benchmark = True

    # build model
    if args.model.lower() == 'pct':
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

    # model = nn.DataParallel(model).cuda()
        
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
    # if args.model.lower() != 'pointtransformer':
    model = DistributedDataParallel(
                model.cuda(), device_ids=[args.local_rank], find_unused_parameters=True)
    # model = model.cuda()
    # prepare data
    if args.dataset=='mn40':
        train_set = ModelNetDataLoader(root=args.data_root, args=args, split='train', process_data=args.process_data)
        test_set = ModelNetDataLoader(root=args.data_root, args=args, split='test', process_data=args.process_data)
        
        train_sampler = DistributedSampler(train_set, shuffle=False)
        test_sampler = DistributedSampler(test_set, shuffle=False)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=True, sampler=train_sampler)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=10, drop_last=False, sampler=test_sampler)
    
    else:
        train_set = ScanObjectNN(partition='training', num_points=args.num_points)
        test_set = ScanObjectNN(partition='test', num_points=args.num_points)
        train_sampler = DistributedSampler(train_set, shuffle=False)
        test_sampler = DistributedSampler(test_set, shuffle=False)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=True, sampler=train_sampler)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=10, drop_last=False, sampler=test_sampler)
    
    ### add methods
    if args.attack_methods=='3d-adv':
         # setup attack settings
        if args.adv_func == 'logits':
            adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa_3d_adv)
        else:
            adv_func = CrossEntropyAdvLoss()
        dist_func = L2Dist()
        clip_func = ClipPointsLinf(budget=args.epsilon)
        # hyper-parameters from their official tensorflow code
        attacker = CWPerturb(args.model.lower(), model, adv_func, dist_func,
                            attack_lr=args.attack_lr_3d_adv,
                            init_weight=10., max_weight=80.,
                            binary_step=args.binary_step_3d_adv,
                            num_iter=args.num_iter_3d_adv, clip_func=clip_func)
    
    elif args.attack_methods=='knn':
         # setup attack settings
        if args.adv_func == 'logits':
            adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa_knn)
        else:
            adv_func = CrossEntropyAdvLoss()
        # hyper-parameters from their official tensorflow code
        dist_func = ChamferkNNDist(chamfer_method='adv2ori',
                                knn_k=5, knn_alpha=1.05,
                                chamfer_weight=5., knn_weight=3.)
        clip_func = ProjectInnerClipLinf(budget=args.epsilon)
        attacker = CWUKNN(args.model.lower(),model, adv_func, dist_func, clip_func,
                        attack_lr=args.attack_lr_knn,
                        num_iter=args.num_iter_knn)
    
    elif args.attack_methods=='advpc':
         #AutoEncoder model
        ae_model = encoders_decoders.AutoEncoder(3)
        ae_state_dict = torch.load(args.ae_model_path, map_location='cpu')
        print('Loading ae weight {}'.format(args.ae_model_path))
        try:
            ae_model.load_state_dict(ae_state_dict)
        except RuntimeError:
            ae_state_dict = {k[7:]: v for k, v in ae_state_dict.items()}
            ae_model.load_state_dict(ae_state_dict)

        ae_model = DistributedDataParallel(
            ae_model.cuda(), device_ids=[args.local_rank])
        # setup attack settings
        if args.adv_func == 'logits':
            adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa_advpc)
        else:
            adv_func = CrossEntropyAdvLoss()
        clip_func = ClipPointsLinf(budget=args.epsilon)
        dist_func = ChamferDist()

        attacker = CWUAdvPC(model, ae_model, adv_func, dist_func,
                            attack_lr=args.attack_lr_advpc,
                            binary_step=args.binary_step_advpc,
                            num_iter=args.num_iter_advpc, GAMMA=args.GAMMA_advpc,
                            clip_func=clip_func)
    
    elif args.attack_methods=='pf-attack':
        # setup attack settings
        if args.adv_func_pf == 'logits':
            adv_func = UntargetedLogitsAdvLoss(kappa=args.kappa)
        else:
            adv_func = AdvCrossEntropyLoss()
        dist_func = ChamferDist()
        clip_func = ClipLinf(budget=args.epsilon)
        # hyper-parameters from their official tensorflow code
        attacker = CW_PFattack(model_name=args.model.lower(), model=model,  adv_func=adv_func,
                            dist_func=dist_func,players=args.players, 
                            k_sharp=args.k_sharp, initial_const = args.initial_const_pf,
                            pp = args.pp_pf, trans_weight = args.trans_weight_pf,
                            attack_lr=args.attack_lr_pf, binary_step=args.binary_step_pf,
                            num_iter=args.num_iter_pf, clip_func=clip_func)
    elif args.attack_methods=='aof':
        # setup attack settings
        if args.adv_func == 'logits':
            adv_func = UntargetedLogitsAdvLoss(kappa=30.)
        else:
            adv_func = CrossEntropyAdvLoss()
        clip_func = ClipPointsLinf(budget=args.epsilon)
        dist_func = L2Dist()
        attacker = CWAOF(model, adv_func, dist_func,
                         attack_lr=args.attack_lr_aof,
                         binary_step=args.binary_step_aof,
                         num_iter=args.num_iter_aof, GAMMA=args.GAMMA_aof,
                         low_pass = args.low_pass_aof,
                         clip_func=clip_func)
    elif args.attack_methods=='ss-aof':
        # setup attack settings
        if args.adv_func == 'logits':
            adv_func = UntargetedLogitsAdvLoss(kappa=30.)
        else:
            adv_func = CrossEntropyAdvLoss()
        clip_func = ClipPointsLinf(budget=args.epsilon)
        dist_func = L2Dist()
        attacker = SSCWAOF(model, adv_func, dist_func,
                         attack_lr=args.attack_lr_aof,
                         binary_step=args.binary_step_aof,
                         num_iter=args.num_iter_aof, GAMMA=args.GAMMA_aof,
                         low_pass = args.low_pass_aof,
                         clip_func=clip_func)
    
    
    def attack(model, attacker, test_loader, args):
        model.eval()
        result = []
        all_ori_pc = []
        all_adv_pc = []
        all_real_lbl = []
        num = 0 #os.path.join(args.save_dir, args.model +'_trans_loss_' + str(args.trans_loss)+'_'+ str(args.lam_trans)+'_div_loss_' + str(args.div_loss) + '_'+ str(args.lam_div) +'_eps_' +str(args.epsilon) + '.pth')
        #os.path.join(args.save_dir, args.model +'_eps_' +str(args.epsilon) + '.pth')
        if os.path.exists(os.path.join(args.save_dir, args.model +'_trans_loss_' + str(args.trans_loss)+'_'+ str(args.lam_trans)+'_div_loss_' + str(args.div_loss) + '_'+ str(args.lam_div) +'_eps_' +str(args.epsilon) + '.pth')):
            print("load model success!!!")
            ### generate attack results
            G = AutoEncoder_z(3)
            state_dict = torch.load(os.path.join(args.save_dir, args.model +'_trans_loss_' + str(args.trans_loss)+'_'+ str(args.lam_trans)+'_div_loss_' + str(args.div_loss) + '_'+ str(args.lam_div) +'_eps_' +str(args.epsilon) + '.pth'),\
                                    map_location='cpu')
            try:
                G.load_state_dict(state_dict)
            except RuntimeError:
                # eliminate 'module.' in keys
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                G.load_state_dict(state_dict)
            print('Loading weight {}'.format(os.path.join(args.save_dir, args.model +'_trans_loss_' + str(args.trans_loss)+'_'+ str(args.lam_trans)+'_div_loss_' + str(args.div_loss) + '_'+ str(args.lam_div) +'_eps_' +str(args.epsilon) + '.pth')))
            G = DistributedDataParallel(
                        G.cuda(), device_ids=[args.local_rank], find_unused_parameters=True)
            # G = G.cuda()
      
            G.eval()
            for data in tqdm(test_loader):
                with torch.no_grad():
                    if len(data)==4:
                        flag=4
                        pc, normal, label, sharply = data
                        pc, normal, label, sharply = pc.float().cuda(non_blocking=True), normal.float().cuda(non_blocking=True),\
                        label.long().cuda(non_blocking=True), sharply.float().cuda(non_blocking=True)
                        double_points = torch.cat((pc, pc), dim=0)
                        double_normal = torch.cat((normal, normal), dim=0)
                        double_label = torch.cat((label, label), dim=0)
                    else:
                        flag=2
                        pc, label = data
                        pc, label = pc.float().cuda(non_blocking=True), label.long().cuda(non_blocking=True)
                        double_points = torch.cat((pc, pc), dim=0)
                        double_label = torch.cat((label, label), dim=0)
                if args.finetune_mode=='noise':
                    z = torch.FloatTensor(pc.shape[0] * 2, args.z_dim).normal_().cuda() #（B*2,16）
                    width = double_points.shape[1]
                    spatial_tile_z = torch.unsqueeze(z, -2).expand(-1, width, -1)
                    adv_noise_ = G(double_points.transpose(1, 2).contiguous(), spatial_tile_z.transpose(1, 2).contiguous())
                else:
                    adv_noise_ = torch.randn(double_points.shape[0], double_points.shape[1], 3).cuda() * 1e-7
                # attack!
                if args.attack_methods=='pf-attack':
                    best_pc, success_num = attacker.attack(double_points, adv_noise_, double_normal if flag==4 else None, double_label)
                elif args.attack_methods=='aof':
                    _, best_pc, success_num = attacker.attack(double_points, adv_noise_, double_label)
                elif args.attack_methods=='ss-aof':
                    _, best_pc, success_num = attacker.attack(double_points, adv_noise_, double_label)
                else:
                    best_pc, success_num = attacker.attack(double_points, adv_noise_, double_label)
                # results
                num += success_num
                all_ori_pc.append(double_points[:pc.shape[0]].detach().cpu().numpy())
                all_adv_pc.append(best_pc[:pc.shape[0]].detach().cpu().numpy())
                all_real_lbl.append(double_label[:pc.shape[0]].detach().cpu().numpy())

            # accumulate results
            all_ori_pc = np.concatenate(all_ori_pc, axis=0)  # [num_data, K, 3]
            all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
            all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
            return all_ori_pc, all_adv_pc, all_real_lbl, num
        
        else:
            adv_func = nn.CrossEntropyLoss(reduction='none')
            adv_func_train = nn.CrossEntropyLoss()
            dist_fuc = ChamferkNNDhusdorfist()
            G = AutoEncoder_z(3)
            G = DistributedDataParallel(
                        G.cuda(), device_ids=[args.local_rank], find_unused_parameters=True)
            # G = G.cuda()
           
            # use Adam optimizer, cosine lr decay
            opt = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=0.00001)
            best_dist = 100000000
            best_acc = 1000000000
            for epoch in range(args.epoch):
                dist_loss = UAE_pretrain.train(G, opt, model, train_set, train_loader, args.batch_size, \
                                            epoch, args.z_dim, args.epsilon, adv_func_train, dist_fuc, \
                                            args.target_layer, args.lam_trans, args.lam_div,args.lam_dis, args.div_loss,args.trans_loss)
                avg_acc, l2_loss = UAE_pretrain.test(G, test_loader, args.z_dim, model, args.epsilon)
                print("AVG ACC:{}".format(avg_acc))
                print("L2 Dist:{}".format(l2_loss))
                if l2_loss <= best_dist and avg_acc <= best_acc:
                    best_dist = l2_loss
                    best_acc = avg_acc
                    if dist.get_rank() == 0:
                        torch.save(G.state_dict(), os.path.join(args.save_dir, args.model +'_trans_loss_' + str(args.trans_loss)+'_'+ str(args.lam_trans)+'_div_loss_' + str(args.div_loss) + '_'+ str(args.lam_div) +'_eps_' +str(args.epsilon) + '.pth'))
            if dist.get_rank() == 0:
                torch.save(G.state_dict(), os.path.join(args.save_dir, args.model +'_trans_loss_' + str(args.trans_loss)+'_'+ str(args.lam_trans)+'_div_loss_' + str(args.div_loss) + '_'+ str(args.lam_div) +'_last_epoch_eps_' +str(args.epsilon) + '.pth'))
            
            ### generate attack results
            G = AutoEncoder_z(3)
            state_dict = torch.load(os.path.join(args.save_dir, args.model +'_trans_loss_' + str(args.trans_loss)+'_'+ str(args.lam_trans)+'_div_loss_' + str(args.div_loss) + '_'+ str(args.lam_div) +'_eps_' +str(args.epsilon) + '.pth'),\
                                    map_location='cpu')
            try:
                G.load_state_dict(state_dict)
            except RuntimeError:
                # eliminate 'module.' in keys
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                G.load_state_dict(state_dict)
            print('Loading weight {}'.format(os.path.join(args.save_dir, args.model +'_trans_loss_' + str(args.trans_loss)+'_'+ str(args.lam_trans)+'_div_loss_' + str(args.div_loss) + '_'+ str(args.lam_div) +'_eps_' +str(args.epsilon) + '.pth')))
            G = DistributedDataParallel(
                        G.cuda(), device_ids=[args.local_rank], find_unused_parameters=True)
            G.eval()
            for data in tqdm(test_loader):
                with torch.no_grad():
                    if len(data)==4:
                        flag=4
                        pc, normal, label, sharply = data
                        pc, normal, label, sharply = pc.float().cuda(non_blocking=True), normal.float().cuda(non_blocking=True),\
                        label.long().cuda(non_blocking=True), sharply.float().cuda(non_blocking=True)
                        double_points = torch.cat((pc, pc), dim=0)
                        double_normal = torch.cat((normal, normal), dim=0)
                        double_label = torch.cat((label, label), dim=0)
                    else:
                        flag=2
                        pc, label = data
                        pc, label = pc.float().cuda(non_blocking=True), label.long().cuda(non_blocking=True)
                        double_points = torch.cat((pc, pc), dim=0)
                        double_label = torch.cat((label, label), dim=0)
                if args.finetune_mode=='noise':
                    z = torch.FloatTensor(pc.shape[0] * 2, args.z_dim).normal_().cuda() #（B*2,16）
                    width = double_points.shape[1]
                    spatial_tile_z = torch.unsqueeze(z, -2).expand(-1, width, -1)
                    adv_noise_ = G(double_points.transpose(1, 2).contiguous(), spatial_tile_z.transpose(1, 2).contiguous()) #相当于重构过程 forward大约3000M
                else:
                    adv_noise_ = torch.randn(double_points.shape[0], double_points.shape[1], 3).cuda() * 1e-7
                # attack!
                if args.attack_methods=='pf-attack':
                    best_pc, success_num = attacker.attack(double_points, adv_noise_, double_normal if args.dataset=='mn40' else None, double_label)
                elif args.attack_methods=='aof':
                    _, best_pc, success_num = attacker.attack(double_points, adv_noise_, double_label)
                elif args.attack_methods=='ss-aof':
                    _, best_pc, success_num = attacker.attack(double_points, adv_noise_, double_label)
                else:
                    best_pc, success_num = attacker.attack(double_points, adv_noise_, double_label)
                # results
                num += success_num
                all_ori_pc.append(double_points[:pc.shape[0]].detach().cpu().numpy())
                all_adv_pc.append(best_pc[:pc.shape[0]].detach().cpu().numpy())
                all_real_lbl.append(double_label[:pc.shape[0]].detach().cpu().numpy())

            # accumulate results
            all_ori_pc = np.concatenate(all_ori_pc, axis=0)  # [num_data, K, 3]
            all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
            all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
            return all_ori_pc, all_adv_pc, all_real_lbl, num
        
    # run attack
    ori_data, attacked_data, real_label, success_num = attack(model, attacker, test_loader, args)

    # accumulate results
    data_num = len(test_set)
    success_rate = float(success_num) / float(data_num)

    if args.attack_methods=='3d-adv':
        save_path = './attack/results/{}/finetune_3d-adv_BT{}_ST{}_transloss_{}_{}_div_loss_{}_{}'.\
                        format(args.dataset, args.binary_step_3d_adv, args.num_iter_3d_adv,args.trans_loss,args.lam_trans,args.div_loss,args.lam_div)
    elif args.attack_methods=='knn':
        save_path = './attack/results/{}/finetune_knn_ST{}_transloss_{}_{}_div_loss_{}_{}'.\
                        format(args.dataset,args.num_iter_knn,args.trans_loss,args.lam_trans,args.div_loss,args.lam_div)
    elif args.attack_methods=='advpc':
        save_path = './attack/results/{}/finetune_advpc_BT{}_ST{}_transloss_{}_{}_div_loss_{}_{}'.\
                        format(args.dataset,args.binary_step_advpc, args.num_iter_advpc,args.trans_loss,args.lam_trans,args.div_loss,args.lam_div)
    elif args.attack_methods=='pf-attack':
        save_path = './attack/results/{}/finetune_pf-attack_BT{}_ST{}_transloss_{}_{}_div_loss_{}_{}'.\
                        format(args.dataset,args.binary_step_pf, args.num_iter_pf,args.trans_loss,args.lam_trans,args.div_loss,args.lam_div)
    elif args.attack_methods=='aof':
        save_path = './attack/results/{}/finetune_AOF_BT{}_ST{}_transloss_{}_{}_div_loss_{}_{}'.\
                        format(args.dataset,args.binary_step_aof, args.num_iter_aof,args.trans_loss,args.lam_trans,args.div_loss,args.lam_div)
    elif args.attack_methods=='ss-aof':
        save_path = './attack/results/{}/finetune_SSAOF_BT{}_ST{}_transloss_{}_{}_div_loss_{}_{}'.\
                        format(args.dataset, args.binary_step_aof, args.num_iter_aof,args.trans_loss,args.lam_trans,args.div_loss,args.lam_div)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = 'UFTAN-{}-{}-success_{:.4f}-rank_{}.npz'.\
        format(args.model, args.epsilon,
            success_rate, args.local_rank)
    np.savez(os.path.join(save_path, save_name),
            ori_pc=ori_data.astype(np.float32),
            test_pc=attacked_data.astype(np.float32),
            test_label=real_label.astype(np.uint8))