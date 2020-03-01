# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 9/11/2019

import numpy as np

def iou(bboxes, candidates, metric = 'origin'):
    """Compute intersection over union.

    Parameters
    ----------
    bboxes : ndarray (N,4)
        A Nx4 matrix of bounding boxes in format `(top left x, top left y, width, height)`.
    candidates : ndarray (M,4)
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    iou: ndarray (N,M)
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bboxes, candidates = np.atleast_2d(bboxes), np.atleast_2d(candidates)
    overlap = np.zeros((bboxes.shape[0], candidates.shape[0]))
    if overlap.size == 0:
        return overlap
    if metric == 'shape':
        bboxes[:, :2] *= 0
        candidates[:, :2] *= 0
    elif metric == 'position':
        bboxes[:, :2] -= bboxes[:, 2:]/2
        candidates[:, :2] -= candidates[:, 2:]/2
    for i, bbox in enumerate(bboxes):
        bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
        candidates_tl = candidates[:, :2]
        candidates_br = candidates[:, :2] + candidates[:, 2:]

        tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                   np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
        br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
                   np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
        wh = np.maximum(0., br - tl)# (N, 2)

        area_intersection = wh.prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_candidates = candidates[:, 2:].prod(axis=1)
        overlap[i, :] = area_intersection / (area_bbox + area_candidates - area_intersection)
    return overlap