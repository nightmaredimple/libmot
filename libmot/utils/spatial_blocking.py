# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 23/12/2019

import time
import numpy as np
from .iou import iou

def iou_blocking(tracks, dets, region_shape):
    """Blocking regions for each tracks

    Parameters
    -----------
    tracks: 2-dim ndarray
        Nx4 matrix of (x,y,w,h)
    dets: 2-dim ndarray
        Mx4 matrix of (x,y,w,h)
    region_shape: Tuple(w,h) or array-like (Nx2)
        region shape for each track

    Returns
    ---------
    blocks: ndarray of boolean(NxM)
        block sets for each track,
    """

    tracks = np.atleast_2d(tracks)
    dets = np.atleast_2d(dets)
    if not isinstance(region_shape, tuple):
        region_shape = np.atleast_2d(region_shape)
    else:
        region_shape = np.array([[region_shape[0], region_shape[1]]])
        region_shape = np.tile(region_shape, (tracks.shape[0], 1))

    centers = tracks[:, :2] + tracks[:, 2:]/2.

    overlap = iou(np.c_[centers - region_shape/2., region_shape], dets)
    keep = overlap > 0
    return keep

if __name__ == '__main__':

    tracks = np.array([[680, 407, 67, 199], [1368, 394, 74, 226], [470, 432, 72, 176]])
    dets = np.array([[664, 410, 70, 180], [450, 410, 50, 150], [1920, 1080, 100, 200]])
    tracks = np.tile(tracks, (10,1))
    dets = np.tile(dets, (10,1))

    time0 = time.time()
    keep = iou_blocking(tracks, dets, tracks[:, 2:])
    time1 = time.time()

    print('tracks nums = %d, detections = %d' % (tracks.shape[0], dets.shape[0]))
    print('  iou_blocking    cost %.4f s' % (time1 - time0))



