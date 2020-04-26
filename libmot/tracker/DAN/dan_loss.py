# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 21/4/2020
# referring to https://github.com/shijieS/SST/blob/master/layer/sst_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-5
class DANLoss(nn.Module):
    def __init__(self, cfg):
        super(DANLoss, self).__init__()
        self.max_object = cfg['datasets']['max_object']

    def forward(self, prediction, labels, mask0, mask1):
        """
        Parameters
        ----------
        prediction: predicted affinity matrix
        labels: gt matrix: [B,1,N+1,N+1]
        mask0: [B,1,N+1]
            [True....True,False...False,True] : [0...Nm-1,...N]
        mask1: [B,1,N+1]
            [True....True,False...False,True] : [0...Nn-1,...N]

        Returns
        -------
        loss_pre: Lf=sum(L1*(-log(A1)))/num(L1)
        loss_next: Lb=sum(L2*(-log(A2)))/num(L2)
        loss_similarity: Lc=L4*|A1-A2|
        loss:
            La=sum(L3*(-log(max(A1,A2))))/num(L3)
            loss=(loss_pre + loss_next + loss_a + loss_similarity)/4.0
        accuray_pre: (rowwise_max_of max(A1,A2)[mask] == rowwise_max_of A1[mask])/num(mask)
        accuracy_next:(colwise_max_of max(A1,A2)[mask] == colwise_max_of A1[mask])/num(mask)
        mean_accuray: (accuracy_pre+accuracy_next)/2.0
        index_pre：

        """

        mask_pre = mask0[:, :, :] # [B,1,N+1]
        mask_next = mask1[:, :, :] # [B,1,N+1]
        mask0 = mask0.unsqueeze(3).repeat(1, 1, 1, self.max_object+1)  # [B,1,N+1,N+1]
        mask1 = mask1.unsqueeze(2).repeat(1, 1, self.max_object+1, 1)  # [B,1,N+1,N+1]
        mask0 = mask0.detach().to(device=mask0.device)
        mask1 = mask1.detach().to(device=mask1.device)

        # get valid mask region
        mask_region = (mask0 * mask1).float()   # the valid position mask [B,1,N+1,N+1]
        mask_region_pre = mask_region.clone()
        mask_region_pre[:, :, self.max_object, :] = 0
        mask_region_next = mask_region.clone()
        mask_region_next[:, :, :, self.max_object] = 0
        mask_region_union = mask_region_pre*mask_region_next

        # get A1, A2, max(A1[:N,:N],A2[:N,:N])
        prediction_pre = F.softmax(mask_region_pre*prediction, dim=3)   # softmax in each row,A1
        prediction_next = F.softmax(mask_region_next*prediction, dim=2)  # softmax in each col,A2
        prediction_all = prediction_pre.clone() # max(A1[:N,:N],A2[:N,:N])
        prediction_all[:, :, :self.max_object, :self.max_object] =\
            torch.max(prediction_pre, prediction_next)[:, :, :self.max_object, :self.max_object]

        # mask labels and get loss
        labels = labels.float()
        labels_pre = mask_region_pre * labels
        labels_next = mask_region_next * labels
        labels_union = mask_region_union * labels
        labels_num = labels.sum().item()
        labels_num_pre = labels_pre.sum().item()
        labels_num_next = labels_next.sum().item()
        labels_num_union = labels_union.sum().item()


        # Lf=sum(L1*(-log(A1)))/num(L1)
        if labels_num_pre != 0:
            loss_pre = - (labels_pre * torch.log(prediction_pre + eps)).sum() / labels_num_pre
        else:
            loss_pre = - (labels_pre * torch.log(prediction_pre + eps)).sum()

        # Lb=sum(L2*(-log(A2)))/num(L2)
        if labels_num_next != 0:
            loss_next = - (labels_next * torch.log(prediction_next + eps)).sum() / labels_num_next
        else:
            loss_next = - (labels_next * torch.log(prediction_next + eps)).sum()

        # La=sum(L3*(-log(max(A1,A2))))/num(L3)
        if labels_num_pre != 0 and labels_num_next != 0:
            loss_assemble = -(labels_pre * torch.log(prediction_all + eps)).sum() / labels_num_pre
        else:
            loss_assemble = -(labels_pre * torch.log(prediction_all + eps)).sum()

        # Lc=L4*|A1-A2|
        if labels_num_union != 0:
            loss_similarity = (labels_union * (torch.abs((1-prediction_pre) - (1-prediction_next)))).sum() / labels_num
        else:
            loss_similarity = (labels_union * (torch.abs((1-prediction_pre) - (1-prediction_next)))).sum()

        # (rowwise_max_of max(A1,A2)[mask] == rowwise_max_of A1[mask])/num(mask)
        _, indexes_ = labels_pre.max(3) # max of each row
        indexes_ = indexes_[:, :, :-1]

        _, indexes_pre = prediction_all.max(3)
        indexes_pre = indexes_pre[:, :, :-1]
        mask_pre_num = mask_pre[:, :, :-1].sum().detach().item() # number of valid targets in pre frame

        if mask_pre_num > 0:
            accuracy_pre = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:,:, :-1]]).float().sum() / mask_pre_num
        else:
            accuracy_pre = (indexes_pre[mask_pre[:, :, :-1]] == indexes_[mask_pre[:, :, :-1]]).float().sum() + 1

        # (colwise_max_of max(A1,A2)[mask] == colwise_max_of A1[mask])/num(mask)
        _, indexes_ = labels_next.max(2) # max of each col
        indexes_ = indexes_[:, :, :-1]
        _, indexes_next = prediction_next.max(2)
        indexes_next = indexes_next[:, :, :-1]
        mask_next_num = mask_next[:, :, :-1].sum().detach().item() # number of valid targets in next frame
        if mask_next_num > 0:
            accuracy_next = (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum() / mask_next_num
        else:
            accuracy_next = (indexes_next[mask_next[:, :, :-1]] == indexes_[mask_next[:, :, :-1]]).float().sum() + 1

        return loss_pre, loss_next, loss_similarity, loss_assemble,\
               (loss_pre + loss_next + loss_assemble + loss_similarity)/4.0, \
               accuracy_pre, accuracy_next, (accuracy_pre + accuracy_next)/2.0, indexes_pre

    def get_loss(self, input, target, mask0, mask1):

        return self.forward(input, target, mask0, mask1)
