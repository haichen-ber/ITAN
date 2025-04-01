import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from baselines.multi_granularity import grad_helper
from baselines.multi_granularity.utils import AverageMeter
from pytorch3d.ops import knn_points, knn_gather


def _compare(output, target, gt, targeted):
    if targeted:
        return output == target
    else:
        return output != gt
    


def offset_proj(offset, ori_pc, ori_normal, project='dir'):
    # offset: shape [b, 3, n], perturbation offset of each point
    # normal: shape [b, 3, n], normal vector of the object
    ori_normal = ori_normal / torch.sqrt(torch.sum(ori_normal ** 2, dim=-1, keepdim=True))
    ori_offset = offset.clone()
    if offset.shape[1] !=3:
        offset = offset.transpose(1, 2).contiguous()
        ori_pc = ori_pc.transpose(1, 2).contiguous()
        ori_normal = ori_normal.transpose(1, 2).contiguous()
    condition_inner = torch.zeros(offset.shape).cuda().byte()

    intra_KNN = knn_points(offset.permute(0,2,1), ori_pc.permute(0,2,1), K=1) #[dists:[b,n,1], idx:[b,n,1]]
    normal = knn_gather(ori_normal.permute(0,2,1), intra_KNN.idx).permute(0,3,1,2).squeeze(3).contiguous() # [b, 3, n]

    normal_len = (normal**2).sum(1, keepdim=True).sqrt()
    normal_len_expand = normal_len.expand_as(offset) #[b, 3, n]

    # add 1e-6 to avoid dividing by zero
    offset_projected = (offset * normal / (normal_len_expand + 1e-6)).sum(1,keepdim=True) * normal / (normal_len_expand + 1e-6)

    # let perturb be the projected ones
    offset = torch.where(condition_inner, offset, offset_projected)
    if offset.shape[1]!=ori_offset.shape[1]:
        offset = offset.transpose(1, 2).contiguous()
    return offset


def scale_points(diff, epsilon):
    diff_ori = diff.clone()
    if diff.shape[1]==3:
        diff_ = diff.clone()
    else:
        diff_ = diff.transpose(1, 2).contiguous()
    with torch.no_grad():
        norm = torch.sum(diff_ ** 2, dim=1) ** 0.5  # [B, K]
        scale_factor = epsilon / (norm + 1e-9)  # [B, K]
        scale_factor = torch.clamp(scale_factor, max=1.)  # [B, K]
        diff_ = diff_ * scale_factor[:, None, :]
    if diff_.shape[1] != diff_ori.shape[1]:
        diff_ = diff_.transpose(1, 2).contiguous()
    return diff_

def renormalization(X, X_pert, epsilon):
    eps_added = scale_points(X_pert - X, epsilon) + X
    return eps_added


def feature_distance_loss(feature1, feature2, alpha):
    t_feature1 = torch.sign(feature1) * torch.pow(torch.abs(feature1+1e-4), alpha)
    t_feature2 = torch.sign(feature2) * torch.pow(torch.abs(feature2+1e-4), alpha)
    dis = torch.norm(t_feature1-t_feature2, 2, dim=1)
    return dis



def train(G, G_optimizer, class_model, dataset, dataloader, batch_size, epoch, z_dim, epsilon, adv_func, \
          dist_fuc, target_layer, lam_trans, lam_div, lam_dis, div_loss_, trans_loss_):
    G.train()
    class_model.eval()
    #############################################################################################################
    # Training
    trans_losses = AverageMeter()
    div_losses = AverageMeter()
    cls_losses = AverageMeter()
    #############################################################################################################
    with tqdm(total=(len(dataset) - len(dataset) % batch_size)) as _tqdm:
        _tqdm.set_description('Epoch: {}/{}'.format(epoch, epoch))
        for data in dataloader:
            if len(data)==3:
                flag=3
                points, normal, label = data
                points, normal, label = points.cuda(), normal.cuda(), label.cuda()
                double_points = torch.cat((points, points), dim=0)
                double_normal = torch.cat((normal, normal), dim=0)
                double_label = torch.cat((label, label), dim=0)
            else:
                flag=2
                points, label = data
                points, label = points.cuda(), label.cuda()
                double_points = torch.cat((points, points), dim=0)
                double_label = torch.cat((label, label), dim=0)
            
            # 第一次 对抗loss
            z = torch.FloatTensor(points.shape[0] * 2, z_dim).normal_().cuda() #（B*2,16）
            width = double_points.shape[1]
            spatial_tile_z = torch.unsqueeze(z, -2).expand(-1, width, -1)
            adv_noise = G(double_points.transpose(1, 2).contiguous(), spatial_tile_z.transpose(1, 2).contiguous())
            if flag==3:
                proj_offset = offset_proj(adv_noise, double_points, double_normal)
                adv_noise.data = proj_offset.data
            double_adv_points = double_points + adv_noise
            double_adv_points.data = renormalization(double_points.data, double_adv_points.data, epsilon)
            ### 距离损失
            dist_loss = dist_fuc(double_adv_points, double_points)
            dist_loss = lam_dis * dist_loss
            G_optimizer.zero_grad()
            dist_loss.backward() 

            adv_noise3 = G(double_points.transpose(1, 2).contiguous(), spatial_tile_z.transpose(1, 2).contiguous())
            
            if flag==3:
                proj_offset3 = offset_proj(adv_noise3, double_points, double_normal)
                adv_noise3.data = proj_offset3.data
            double_adv_points3 = double_points + adv_noise3
            double_adv_points3.data = renormalization(double_points.data, double_adv_points3.data, epsilon)
            adv_output = class_model(double_adv_points3.transpose(1, 2).contiguous())[0]

            ## 对抗损失
            cls_loss = -adv_func(adv_output, double_label.long().cuda().detach())

            cls_loss.backward()
            
            if trans_loss_:
                ### 第二次 迁移性loss
                ori_gcam = grad_helper.GradCAM(model=class_model, candidate_layers=target_layer)
                output = ori_gcam.forward(double_points)
                ori_gcam.backward(logits=output, targets=double_label.long().cuda().detach())
                feature_map_ori, att = ori_gcam.generate(target_layer=target_layer)

                ## 重新输入
                adv_noise1 = G(double_points.transpose(1, 2).contiguous(), spatial_tile_z.transpose(1, 2).contiguous()) 
                if flag==3:
                    proj_offset1 = offset_proj(adv_noise1, double_points, double_normal)
                    adv_noise1.data = proj_offset1.data
                double_adv_points1 = double_points + adv_noise1
                double_adv_points1.data = renormalization(double_points.data, double_adv_points1.data, epsilon)
                
                adv_gcam = grad_helper.GradCAM(model=class_model, candidate_layers=target_layer)
                adv_output1 = adv_gcam.forward(double_adv_points1)
                adv_gcam.backward(logits=adv_output1, targets=double_label.long().cuda().detach())
                feature_map_adv, att1 = adv_gcam.generate(target_layer=target_layer)
                trans_loss = lam_trans * ((att * feature_map_adv)).sum()
                trans_loss.backward()
            if div_loss_:
                ###第三次 kl loss
                adv_noise2 = G(double_points.transpose(1, 2).contiguous(), spatial_tile_z.transpose(1, 2).contiguous())
                if flag==3:
                    proj_offset2 = offset_proj(adv_noise2, double_points, double_normal)
                    adv_noise2.data = proj_offset2.data
                double_adv_points2 = double_points + adv_noise2
                double_adv_points2.data = renormalization(double_points.data, double_adv_points2.data, epsilon)
                adv_gcam1 = grad_helper.GradCAM(model=class_model, candidate_layers=target_layer)
                adv_output2 = adv_gcam1.forward(double_adv_points2)
                adv_gcam1.backward(logits=adv_output2, targets=double_label.long().cuda().detach())
                feature_map_adv2, att2 = adv_gcam1.generate(target_layer=target_layer)

                div_loss = feature_distance_loss(feature_map_adv2[:points.shape[0]], feature_map_adv2[points.shape[0]:], 0.5)
                    
                div_loss = -1 * lam_div * div_loss.sum()
                div_loss.backward()
            G_optimizer.step()
            
            if trans_loss_:
                ori_gcam.remove_hook()
                ori_gcam.clear()
                adv_gcam.remove_hook()
                adv_gcam.clear() 
            if div_loss_:
                adv_gcam1.remove_hook()
                adv_gcam1.clear()       

            cls_losses.update(cls_loss.data, len(points) * 2)
            if trans_loss_:
                trans_losses.update(trans_loss.data, len(points))
            if div_loss_:
                div_losses.update(div_loss.data, len(points))

            if div_loss_ and trans_loss_:
                _tqdm.set_postfix(
                    attn_loss='{:.4f}'.format(trans_losses.avg),
                    cls_loss='{:.4f}'.format(cls_losses.avg),
                    div_loss='{:.4f}'.format(div_losses.avg),
                )
            else:
                if trans_loss_:
                    _tqdm.set_postfix(
                        attn_loss='{:.4f}'.format(trans_losses.avg),
                        cls_loss='{:.4f}'.format(cls_losses.avg),
                    )
                if div_loss_:
                    _tqdm.set_postfix(
                        cls_loss='{:.4f}'.format(cls_losses.avg),
                        div_loss='{:.4f}'.format(div_losses.avg),
                    )
            _tqdm.update(batch_size)


def test(G, testloader, z_dim, class_model, epsilon):
    G.eval()
    class_model.eval()
    correct = 0
    total = 0
    l2_loss_ = 0
    with torch.no_grad():
        for data in testloader:
            if len(data)==3:
                flag=3
                points, normal, label = data
                points, normal, label = points.cuda(), normal.cuda(), label.cuda()
                double_points = torch.cat((points, points), dim=0)
                double_normal = torch.cat((normal, normal), dim=0)
                double_label = torch.cat((label, label), dim=0)
            else:
                flag=2
                points, label = data
                points, label = points.cuda(), label.cuda()
                double_points = torch.cat((points, points), dim=0)
                double_label = torch.cat((label, label), dim=0)
            z = torch.FloatTensor(points.shape[0] * 2, z_dim).normal_().cuda()
            width = double_points.shape[1]
            spatial_tile_z = torch.unsqueeze(z, -2).expand(-1, width, -1)
            adv_noise = G(double_points.transpose(1, 2).contiguous(), spatial_tile_z.transpose(1, 2).contiguous())
            if flag==3:
                proj_offset = offset_proj(adv_noise, double_points, double_normal)
                adv_noise.data = proj_offset.data
            double_adv_points = double_points + adv_noise
            double_adv_points.data = renormalization(double_points.data, double_adv_points.data, epsilon)
            l2_loss =  torch.sqrt(torch.sum((double_adv_points - double_points) ** 2, dim=[1, 2]))
            output = class_model(double_adv_points.transpose(1, 2).contiguous())[0]
            _, predicted = torch.max(output.data, 1)
            total += double_label.size(0)
            correct += (predicted == double_label).sum().item()
            l2_loss_ +=l2_loss.sum()
    avg_acc = 100 * float(correct) / total
    l2_loss = l2_loss_.sum() / total
    return avg_acc, l2_loss