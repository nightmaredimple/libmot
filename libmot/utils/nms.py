# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 14/3/2020

import torch

def nms_torch(bboxes, scores, threshold=0.5):
    """Non Maximum Suppression

    Parameters
    ----------
    bboxes : tensor (N,4)
        A Nx4 matrix of bounding boxes in format `(top left x, top left y, bottom right x, bottom right y)`.
    scores : tensor (N,)
        A vector of detection scores for bboxes.
    threshold: float
        threshold of iou
    Returns
    -------
    keep : tensor (K, )
        A vector of indexes for remained bboxes.
    """
    area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:

        if order.numel() == 1:
            keep.append(order.item())
            break
        else:
            i = order[0].item()
            keep.append(i)
        # compute iou ,[N-1]
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)

        iou = inter / (area[i] + area[order[1:]] - inter)
        idx = (iou <= threshold).nonzero().squeeze()  # Notice the idx+1 -> order
        if idx.numel() == 0:
            break
        order = order[idx + 1]

    return torch.LongTensor(keep)
