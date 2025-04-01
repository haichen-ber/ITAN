"""Test the victim models"""
import argparse
import numpy as np
import os
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
sys.path.append('./')
from baselines.dataset import ModelNet40Attack, ModelNet40Transfer, load_data
from baselines.model import DGCNN, PointNetCls, PointNet2ClsSsg, PointConvDensityClsSsg, Pct, PointTransformerCls
from baselines.util.utils import AverageMeter, str2bool, set_seed
from baselines.config import BEST_WEIGHTS
from baselines.config import MAX_TEST_BATCH as BATCH_SIZE
from baselines.config import MAX_DUP_TEST_BATCH as DUP_BATCH_SIZE
from baselines.attack import L2Dist, ChamferDist, HausdorffDist
import pickle


def batched_triangle_indices(distances, k):
    """
    Generate triangle indices for each point cloud in the batch.
    """
    _, indices = torch.topk(distances, k=k+1, largest=False)
    B, N, _ = indices.shape
    triangles = []
    for j in range(1, k + 1):
        triangles.append(torch.stack([torch.arange(N).unsqueeze(0).repeat(B, 1).cuda(), indices[:, :, j], indices[:, :, (j % k) + 1]], dim=-1))
    return torch.cat(triangles, dim=1)


def batched_cdist(a, b):
    """
    Compute pairwise distance between each pair of points in a and b in a batched manner.
    """
    diff = a.unsqueeze(2) - b.unsqueeze(1)
    return torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-9)


def write_ply(save_path, points,text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]  #[batchsize,1024,3]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)
    

def load_pickle(file_name):
	f = open(file_name, "rb+")
	data = pickle.load(f)
	f.close()
	return data


def merge(data_root, prefix):
    ori_data_lst = []
    adv_data_lst = []
    label_lst = []
    save_name = prefix+"-concat.npz"
    if os.path.exists(os.path.join(data_root, save_name)):
        return os.path.join(data_root, save_name)
    for file in os.listdir(data_root):
        if file.startswith(prefix):
            file_path = os.path.join(data_root, file)
            ori_data, adv_data, label = load_data(file_path, partition='transfer')
            # ori_data, adv_data, label = load_pickle(file_path)
            ori_data_lst.append(ori_data)
            adv_data_lst.append(adv_data)
            label_lst.append(label)
    all_ori_pc = np.concatenate(ori_data_lst, axis=0)  # [num_data, K, 3]
    all_adv_pc = np.concatenate(adv_data_lst, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(label_lst, axis=0)  # [num_data]

    np.savez(os.path.join(data_root, save_name),
             ori_pc=all_ori_pc.astype(np.float32),
             test_pc=all_adv_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.uint8))
    return os.path.join(data_root, save_name)

def merge_attack(data_root, prefix):
    target_label_lst = []
    adv_data_lst = []
    label_lst = []
    save_name = prefix+"-concat.npz"
    if os.path.exists(os.path.join(data_root, save_name)):
        return os.path.join(data_root, save_name)
    for file in os.listdir(data_root):
        if file.startswith(prefix):
            file_path = os.path.join(data_root, file)
            adv_data, label, target_label = \
                load_data(file_path, partition='attack')
            adv_data_lst.append(adv_data)
            label_lst.append(label)
            target_label_lst.append(target_label)
    all_adv_pc = np.concatenate(adv_data_lst, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(label_lst, axis=0)  # [num_data]
    all_target_lbl = np.concatenate(target_label_lst, axis=0)  # [num_data]

    np.savez(os.path.join(data_root, save_name),
             test_pc=all_adv_pc.astype(np.float32),
             test_label=all_real_lbl.astype(np.uint8),
             target_label=all_target_lbl.astype(np.uint8))
    return os.path.join(data_root, save_name)


def get_model_name(npz_path):
    """Get the victim model name from npz file path."""
    if 'dgcnn' in npz_path.lower():
        return 'dgcnn'
    if 'pointconv' in npz_path.lower():
        return 'pointconv'
    if 'pointnet2' in npz_path.lower():
        return 'pointnet2'
    if 'pointnet' in npz_path.lower():
        return 'pointnet'
    print('Victim model not recognized!')
    exit(-1)


def test_target():
    """Target test mode.
    Show both classification accuracy and target success rate.
    """
    model.eval()
    acc_save = AverageMeter()
    success_save = AverageMeter()
    with torch.no_grad():
        for data, label, target in tqdm(test_loader):
            data, label, target = \
                data.float().cuda(), label.long().cuda(), target.long().cuda()
            # to [B, 3, N] point cloud
            data = data.transpose(1, 2).contiguous()
            batch_size = label.size(0)
            # batch in
            if args.model.lower() == 'pointnet':
                logits = model(data)[0]
            else:
                logits = model(data)[0]
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == label).sum().float() / float(batch_size)
            acc_save.update(acc.item(), batch_size)
            success = (preds == target).sum().float() / float(batch_size)
            success_save.update(success.item(), batch_size)

    print('Overall accuracy: {:.4f}, '
          'attack success rate: {:.4f}'.
          format(acc_save.avg, success_save.avg))


def test_normal():
    """Normal test mode.
    Test on all data.
    """
    model.eval()
    at_num, at_denom = 0, 0
    l2_dist = 0
    chamfer_dist=0
    hausdorf_dist=0
    num, denom = 0, 0
    num_error = 0
    with torch.no_grad():
        for ori_data, adv_data, label in test_loader:
            ori_data, adv_data, label = \
                ori_data.float().cuda(), adv_data.float().cuda(), label.long().cuda()
            # if num==0:     
            #     # Batched distance calculation
            #     batch_distances = batched_cdist(ori_data, ori_data)
            #     k = 3

            #     # Batched triangle generation
            #     batch_triangles = batched_triangle_indices(batch_distances, k)
            #     #  Convert the point cloud data to a numpy array for visualization
            #     point_cloud_np = ori_data[0].cpu().numpy()
            #     point_cloud_adv_np = adv_data[0].cpu().numpy()
            #     import trimesh
            #     # Create a mesh using the generated triangles and point cloud
            #     mesh = trimesh.Trimesh(vertices=point_cloud_np, faces=batch_triangles[0].cpu())
            #     mesh_adv = trimesh.Trimesh(vertices=point_cloud_adv_np, faces=batch_triangles[0].cpu())
            #     # Save the mesh to a PLY file
            #     mesh.export('point_cloud_mesh.ply')
            #     mesh_adv.export('point_cloud_adv_mesh.ply')
           
            # l2dist = L2Dist()(adv_data, ori_data, batch_avg=False)
            # chdist = ChamferDist()(adv_data, ori_data, batch_avg=False)
            # hsdist = HausdorffDist()(adv_data, ori_data, batch_avg=False)
            # l2_dist += l2dist.sum()
            # chamfer_dist += chdist.sum()
            # hausdorf_dist += hsdist.sum()
            
            # to [B, 3, N] point cloud
            ori_data = ori_data.transpose(1, 2).contiguous()
            # if num==0:
            #     points = ori_data[0].transpose(0, 1).contiguous()
            #     write_ply('vision/clean_point.ply', points)
            adv_data = adv_data.transpose(1, 2).contiguous()
            # if num==0:
            #     adv_points = adv_data[0].transpose(0, 1).contiguous()
            #     write_ply('vision/adv_point.ply', adv_points)
            batch_size = label.size(0)

            if args.model.lower() == 'pointtransformer':
                ori_data = ori_data.transpose(1, 2).contiguous()
                adv_data = adv_data.transpose(1, 2).contiguous()
            # batch in
            if args.model.lower() == 'pointnet':
                logits = model(ori_data)[0]
                adv_logits = model(adv_data)[0]
            else:
                logits = model(ori_data)[0]
                adv_logits = model(adv_data)[0]
            ori_preds = torch.argmax(logits, dim=-1)
            adv_preds = torch.argmax(adv_logits, dim=-1)
            mask_ori = (ori_preds == label) # 1 1 0 1 0 1
            mask_adv = (adv_preds == label) # 0 1 1 0 1 0
            err_num = (adv_preds!=label)
            at_denom += mask_ori.sum().float().item() #分类成功
            at_num += mask_ori.sum().float().item() - (mask_ori * mask_adv).sum().float().item() #分类不成功的
            denom += float(batch_size)
            num_error += err_num.sum().float().item()
            num += mask_adv.sum().float()

    print('Overall attack success rate: {:.4f}'.format(at_num / (at_denom + 1e-9)))  
    # print('Overall accuracy: {:.4f}'.format(num / (denom + 1e-9)))
    print('Overall accuracy: {:.4f}'.format(at_denom / (denom + 1e-9))) #模型本身的分类成功率
    print('top-1 error: {:.4f}'.format(num_error / (denom + 1e-9)) )
    print('Overall L2 dist: {:.4f}'.format(l2_dist))  
    # print('Overall accuracy: {:.4f}'.format(num / (denom + 1e-9)))
    print('Overall CHamfer dist: {:.4f}'.format(chamfer_dist)) #模型本身的分类成功率
    print('Overall Hausdorf dist: {:.4f}'.format(hausdorf_dist) )


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='attack/results/pointnet_feature2/sharply_feature_attack')
    parser.add_argument('--prefix', type=str,
                        default='Usharply_feature_attack-pointnet-0.45-success')
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['normal', 'target'],
                        help='Testing mode')
    parser.add_argument('--batch_size', type=int, default=100, metavar='BS',
                        help='Size of batch, use config if not specified')
    parser.add_argument('--model', type=str, default='pointnet', metavar='MODEL',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv', 'pointtransformer', 'pct'],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]. '
                             'If not specified, judge from data_root')
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
    parser.add_argument('--dataset', type=str, default='mn40', metavar='N',
                        choices=['mn40', 'remesh_mn40',
                                 'opt_mn40', 'conv_opt_mn40', 'aug_mn40','ori_mn40'])
    parser.add_argument('--normalize_pc', type=str2bool, default=False,
                        help='normalize in dataloader')

    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--feature_transform', type=str2bool, default=True,
                        help='whether to use STN on features in PointNet')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='',
                        help='Model weight to load, use config if not specified')


    parser.add_argument('--num_point', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    args = parser.parse_args()

    # victim model
    if not args.model:
        args.model = get_model_name(args.data_root)

    # random seed
    set_seed(1)

    # in case adding attack
    if 'add' in args.data_root.lower():
        # we add 512 points in adding attacks
        if args.num_points == 1024:
            num_points = 1024
            args.num_points = 1024 + 512
        elif args.num_points == 1024 + 512:
            num_points = 1024
    elif 'cluster' in args.data_root.lower():
        # we add 3*32=96 points in adding cluster attack
        if args.num_points == 1024:
            num_points = 1024
            args.num_points = 1024 + 3 * 32
        elif args.num_points == 1024 + 3 * 32:
            num_points = 1024
    elif 'object' in args.data_root.lower():
        # we add 3*64=192 points in adding object attack
        if args.num_points == 1024:
            num_points = 1024
            args.num_points = 1024 + 3 * 64
        elif args.num_points == 1024 + 3 * 64:
            num_points = 1024
    else:
        num_points = args.num_points

    # determine the weight to use
    BEST_WEIGHTS = BEST_WEIGHTS[args.dataset][num_points]
    BATCH_SIZE = BATCH_SIZE[num_points]
    DUP_BATCH_SIZE = DUP_BATCH_SIZE[num_points]
    if args.batch_size == -1:  # automatic assign
        args.batch_size = BATCH_SIZE[args.model]
    # add point attack has more points in each point cloud
    if 'ADD' in args.data_root:
        args.batch_size = int(args.batch_size / 1.5)
    # sor processed point cloud has different points in each
    # so batch size only can be 1
    if 'sor' in args.data_root:
        args.batch_size = 1

    # enable cudnn benchmark
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

    model = nn.DataParallel(model).cuda()
    # model = model.cuda()

    # load model weight
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path))
    else:
        if args.model.lower() == 'pointtransformer':
            state_dict = torch.load(
            BEST_WEIGHTS[args.model], map_location='cpu')
            # concat 'module.' in keys
            state_dict = {'module.'+k: v for k, v in state_dict['model_state_dict'].items()}
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(BEST_WEIGHTS[args.model], map_location='cuda:0'))

    # prepare data
    if args.mode == 'target':
        data_path = merge_attack(args.data_root, args.prefix)
        test_set = ModelNet40Attack(data_path, num_points=args.num_points,
                                    normalize=args.normalize_pc)
    else:
        data_path = merge(args.data_root, args.prefix)
        test_set = ModelNet40Transfer(data_path, num_points=args.num_point)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=8,
                             pin_memory=True, drop_last=False)

    # test
    if args.mode == 'normal':
        test_normal()
    else:
        test_target()
