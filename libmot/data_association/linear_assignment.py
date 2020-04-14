# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 9/11/2019

import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

inf_cost = 1e+5

def LinearAssignment(cost, threshold = None, method = 'KM'):
    """Using Hungarian or KM algorithm to make linear assignment

    Parameters
    ----------
    cost : ndarray
        A NxM matrix for costs between each track_ids with dection_ids
    threshold: float
        if cost > threshold, then will not be considered
    method : str
        'KM': weighted assignment
        'Hungarian': 01 assignment

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
    min_cost: float
        cost of assignments
    """
    cost_c = deepcopy(np.atleast_2d(cost))
    sz = cost_c.shape

    if threshold is not None:
        cost_c = np.where(cost_c > threshold, inf_cost, cost_c)

    if method == 'Hungarian':
        t = threshold if threshold is not None else inf_cost
        cost_c = np.where(cost_c < t, 0, cost_c)

    # linear assignment
    row_ind, col_ind = linear_sum_assignment(cost_c)
    if threshold is not None:
        t = inf_cost - 1 if threshold == inf_cost else threshold
        mask = cost_c[row_ind, col_ind] <= t
        row_idx = row_ind[mask]
        col_idx = col_ind[mask]
    else:
        row_idx, col_idx = row_ind, col_ind

    unmatched_rows = np.array(list(set(range(sz[0])) - set(row_idx)))
    unmatched_cols = np.array(list(set(range(sz[1])) - set(col_idx)))

    min_cost = cost[row_idx, col_idx].sum()

    return row_idx, col_idx, np.sort(unmatched_rows), np.sort(unmatched_cols), min_cost



