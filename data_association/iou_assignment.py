# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 9/11/2019

import numpy as np


def GreedyAssignment(cost, threshold = None, method = 'global'):
    """Using iou matching to make linear assignment

    Parameters
    ----------
    cost : ndarray
        A NxM matrix for costs between each track_ids with dection_ids
    threshold: float
        if cost > threshold, then will not be considered
    method: str
        eg: global, local

    Returns
    -------
    row_idx: List of matched tracks (<=N,)
        assigned tracklets' id
    col_idx: List of matched dets (<=M,)
        assigned dets' id
    unmatched_rows: List of unmatched tracks
        unassigned tracklets' id
    unmatched_cols: List of unmatched dets
        unassigned dets' id
    """
    cost_c = np.atleast_2d(cost)
    sz = cost_c.shape

    if threshold is None:
        threshold = 1.0

    row_idx = []
    col_idx = []
    if method == 'global':
        vector_in = list(range(sz[0]))
        vector_out = list(range(sz[1]))
        while min(len(vector_in), len(vector_out)) > 0:
            v = cost_c[np.ix_(vector_in, vector_out)]
            min_cost = np.min(v)

            if min_cost <= threshold:
                place = np.where(v == min_cost)
                row_idx.append(vector_in[place[0][0]])
                col_idx.append(vector_out[place[1][0]])
                del vector_in[place[0][0]]
                del vector_out[place[1][0]]
            else:
                break
    else:
        vector_in = []
        vector_out = list(range(sz[1]))
        index = 0
        while min(sz[0] - len(vector_in), len(vector_out)) > 0:
            if index >= sz[0]:
                break
            place = np.argmin(cost_c[np.ix_([index], vector_out)])

            if cost_c[index, vector_out[place]] <= threshold:
                row_idx.append(index)
                col_idx.append(vector_out[place])
                del vector_out[place]
            else:
                vector_in.append(index)
            index += 1
        vector_in += list(range(index, sz[0]))

    return np.array(row_idx), np.array(col_idx), np.array(vector_in), np.array(vector_out)