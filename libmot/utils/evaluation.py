# -*- coding: utf-8 -*-
# Author : hongweiwang
# Email  : hongweiwang@hust.edu.cn
# Date   : 13/12/2019

import motmetrics as mm
import pandas as pd
import numpy as np

def evaluation_mot(groundtruths, tracks):
    """
    Compute metrics for trackers using MOTChallenge ground-truth data.
    Parameters
    ----------
    groundtruths:
        ground truth
    tracks :
        tracking result

    """
    assert isinstance(tracks, list) or isinstance(tracks, np.ndarray), 'inputs must be array'

    gts = pd.DataFrame(groundtruths[:, :6], columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
    gts = gts.set_index(['FrameId', 'Id'])
    gts[['X', 'Y']] -= (1, 1)

    box = pd.DataFrame(np.array(tracks[:, :6]), columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
    box = box.set_index(['FrameId', 'Id'])
    box[['X', 'Y']] -= (1, 1)

    acc = mm.utils.compare_to_groundtruth(gts, box, 'iou', distth=0.5)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, \
                         return_dataframe=False)

    return abs(summary['mota']), abs(summary['motp']), abs(summary['idf1']), abs(summary['num_switches'])








