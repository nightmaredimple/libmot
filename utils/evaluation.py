# -*- coding: utf-8 -*-
# Author : hongweiwang
# Email  : hongweiwang@hust.edu.cn
# Date   : 13/12/2019

import glob
import os
import motmetrics as mm
import pandas as pd
import numpy as np
from collections import OrderedDict
from pathlib import Path


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
    return accs, names


def evaluation_mot(groundtruths, tracks, fmt='mot15-2D'):
    """
    Compute metrics for trackers using MOTChallenge ground-truth data.
    Parameters
    ----------
    groundtruths:
        ground truth file.
        or an array of ground truth
    tracks :
        Directory containing tracker result files
        or one tracker result file
        or an array of tracker result
    fmt:str
        Data format

    """
    if isinstance(tracks, list) or isinstance(tracks, np.ndarray):
        gts = pd.DataFrame(groundtruths[:, :6], columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
        gts = gts.set_index(['FrameId', 'Id'])
        gts[['X', 'Y']] -= (1, 1)

        box = pd.DataFrame(np.array(tracks), columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
        box = box.set_index(['FrameId', 'Id'])
        box[['X', 'Y']] -= (1, 1)

        acc = mm.utils.compare_to_groundtruth(gts, box, 'iou', distth=0.5)

        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, \
                             return_dataframe=False)

        return abs(summary['mota']), abs(summary['motp']), abs(summary['idf1']), abs(summary['num_switches'])

    if os.path.isdir(tracks):
        tsfiles = [f for f in glob.glob(os.path.join(tracks, '*.txt')) if not os.path.basename(f).startswith('eval')]
    elif os.path.isfile(tracks):
        tsfiles = [tracks]

    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt)) for f in tsfiles])

    if os.path.isdir(groundtruths):
        gtfiles = glob.glob(os.path.join(groundtruths, '*/gt/gt.txt'))
        gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt, min_confidence=1)) for f in gtfiles])
    else:
        gt = OrderedDict([(Path(groundtruths).parts[-3], mm.io.loadtxt(groundtruths, fmt, min_confidence=1))])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))


if __name__ == '__main__':
    evaluation_mot(groundtruths=r'E:\\datasets\\MOT17\\train\\MOT17-10-SDP\\gt\\gt.txt',\
                   tracks = r'E:\\datasets\\MOT17\\train\\MOT17-10-SDP\\det\\det.txt')






