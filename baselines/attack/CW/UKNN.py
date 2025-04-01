"""Implementation of optimization based attack,
    CW Attack for ROBUST point perturbation.
Based on AAAI'20: Robust Adversarial Objects against Deep Learning Models.
"""

import pdb
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class CWUKNN:
    """Class for CW attack.
    """

    def __init__(self, model_name, model, adv_func, dist_func, clip_func,
                 attack_lr=1e-3, num_iter=2500):
        """CW attack by kNN attack.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            dist_func (function): distance metric
            clip_func (function): clipping function
            attack_lr (float, optional): lr for optimization. Defaults to 1e-3.
            num_iter (int, optional): max iter num in every search step. Defaults to 2500.
        """

        self.model = model.cuda()
        self.model.eval()
        self.model_name = model_name
        self.adv_func = adv_func
        self.dist_func = dist_func
        self.clip_func = clip_func
        self.attack_lr = attack_lr
        self.num_iter = num_iter

    def attack(self, data, noise, target):
        """Attack on given data to target.

        Args:
            data (torch.FloatTensor): victim data, [B, num_points, 3]
            target (torch.LongTensor): target output, [B]
        """
        B, K = data.shape[:2]
        data = data.float().cuda().detach()
        if self.model_name !=  'pointtransformer':
            data = data.transpose(1, 2).contiguous()
        ori_data = data.clone().detach()
        ori_data.requires_grad = False

        # points and normals
        normal = None
        
        target = target.long().cuda().detach()
        label_val = target.detach().cpu().numpy()  # [B]
        # init variables with small perturbation
        if self.model_name !=  'pointtransformer':
            if noise is not None:
                noise1 = noise.transpose(1, 2).detach().clone().cuda()
            else:
                noise1 = torch.randn((B, 3, K)).cuda() * 1e-7
        else:
            if noise is not None:
                noise1 = noise.detach().clone().cuda()
            else:
                noise1 = torch.randn((B, K, 3)).cuda() * 1e-7
       
        adv_data = ori_data.clone().detach() + noise1
        
        adv_data.requires_grad_()
        opt = optim.Adam([adv_data], lr=self.attack_lr, weight_decay=0.)

        adv_loss = torch.tensor(0.).cuda()
        dist_loss = torch.tensor(0.).cuda()

        total_time = 0.
        forward_time = 0.
        backward_time = 0.
        clip_time = 0.

        # record best results in binary search
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = torch.zeros(B, 3, K).to(data.device)

        # there is no binary search in this attack
        # just longer iterations of optimization
        per_batch_time = time.time()
        for iteration in range(self.num_iter):
            t1 = time.time()

            # forward passing
            logits = self.model(adv_data)  # [B, num_classes]
            if isinstance(logits, tuple):  # PointNet
                logits = logits[0]

            t2 = time.time()
            forward_time += t2 - t1

            # print
            pred = torch.argmax(logits, dim=1)  # [B]
            success_num = (pred != target).sum().item()
            if iteration % (self.num_iter // 5) == 0:
                print('Iteration {}/{}, success {}/{}\n'
                      'adv_loss: {:.4f}, dist_loss: {:.4f}'.
                      format(iteration, self.num_iter, success_num, B,
                             adv_loss.item(), dist_loss.item()))

            # compute loss and backward
            adv_loss = self.adv_func(logits, target).mean()

            # in the official tensorflow code, they use sum instead of mean
            # so we multiply num_points as sum
            if self.model_name !=  'pointtransformer':
                dist_loss = self.dist_func(
                    adv_data.transpose(1, 2).contiguous(),
                    ori_data.transpose(1, 2).contiguous()).mean() * K
            else:
                dist_loss = self.dist_func(
                    adv_data,
                    ori_data).mean() * K

            loss = adv_loss + dist_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            t3 = time.time()
            backward_time += t3 - t2

            # clipping and projection!
            adv_data.data = self.clip_func(adv_data.clone().detach(),
                                           ori_data, normal)

            t4 = time.time()
            clip_time = t4 - t3
            total_time += t4 - t1

            if iteration % 100 == 0:
                print('total time: {:.2f}, for: {:.2f}, '
                      'back: {:.2f}, clip: {:.2f}'.
                      format(total_time, forward_time,
                             backward_time, clip_time))
                total_time = 0.
                forward_time = 0.
                backward_time = 0.
                clip_time = 0.
                torch.cuda.empty_cache()
            # record values!
            # forward passing
            with torch.no_grad():
                logits = self.model(adv_data)  # [B, num_classes]
                if isinstance(logits, tuple):  # PointNet
                    logits = logits[0]
                pred = torch.argmax(logits, dim=1)  # [B]
            dist_val = torch.sqrt(torch.sum(
                (adv_data - ori_data) ** 2, dim=[1, 2])).\
                detach().cpu().numpy()  # [B]
            pred_val = pred.detach().cpu().numpy()  # [B]
            input_val = adv_data  # [B, 3, K]

            # update
            for e, (dist, pred, label, ii) in enumerate(zip(dist_val, pred_val, label_val, input_val)):
                if dist < o_bestdist[e] and pred != label:
                    o_bestdist[e] = dist
                    o_bestscore[e] = pred
                    o_bestattack[e] = ii
            if iteration == self.num_iter -1:
                t_batch = time.time()
                total_batch_time = t_batch - per_batch_time
                print('knn total time: {:.2f}'.format(total_batch_time))
                break
        # end of CW attack
        
        # end of CW attack
        # fail to attack some examples
        fail_idx = (o_bestscore < 0)
        o_bestattack[fail_idx] = input_val[fail_idx]
        adv_pc = torch.tensor(o_bestattack).to(adv_data)
        adv_pc = self.clip_func(adv_pc, ori_data)
        with torch.no_grad():
            logits = self.model(adv_pc)  # [B, num_classes]
            if isinstance(logits, tuple):  # PointNet
                logits = logits[0]
            pred = torch.argmax(logits, dim=-1)  # [B]
            success_num = (pred != target).\
                sum().detach().cpu().item()

        # return final results
        print('Successfully attack {}/{}'.format(success_num, B))

        # in their implementation, they estimate the normal of adv_pc
        # we don't do so here because it's useless in our task
        # adv_data = adv_data.transpose(1, 2).contiguous()  # [B, K, 3]
        # adv_data = adv_data.detach().cpu().numpy()  # [B, K, 3]
        return adv_pc, success_num
