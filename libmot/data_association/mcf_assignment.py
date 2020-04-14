# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 11/11/2019
# Update : 11/13/2019

import numpy as np
from ortools.graph import pywrapgraph
from copy import deepcopy

inf_cost = 1e+5
multi_factor = 1e+5


def Solve(Graph, source, sink, n):
    """Solve MinCostFlow with n flows

    Parameters:
    ------------
    source: int
        source node in graph
    sink: int
        sink node in graph
    n: int
        numbers of flows

    Returns:
    ------------
    isOptimal: bool
        whether the Graph is feasible
    """
    Graph.SetNodeSupply(int(source), n)
    Graph.SetNodeSupply(int(sink), -n)
    return Graph.Solve() == Graph.OPTIMAL


def MCFAssignment(cost, threshold):
    """Using MinCostFlow to make Linear assignment between 2 frames
       same to LinearAssignment

    Parameters
    ----------
    cost: Lists[Lists]
        costs between each track_ids with dection_ids
    threshold: float
        if cost > threshold, then will not be considered
    start_nodes: array like
        A (N,) matrix of start node in the graph
    end_nodes: Lists[Lists]
        N Lists of end nodes in the graph
    sparse: bool
        if defined, cost will be correspond with end_notes,
        else start_node and end_nodes will be useless

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
    start_sets, end_sets, Graph = BuildGraphFromDense(cost, threshold)

    source = 0
    sink = max(end_sets) + 1
    row_idx = []
    col_idx = []

    # use binary search to find the most flows
    low = 1
    high = min(len(start_sets), len(end_sets))
    if high < 1:
        min_cost = np.inf
    else:
        record = np.array([False]*high)
        n = 0
        while low <= high:
            mid = (low + high) // 2
            f_high = Solve(Graph, source, sink, high) if not record[high - 1] else record[high - 1]
            record[high - 1] = f_high
            if f_high:
                n = high
                break

            f_low = Solve(Graph, source, sink, low) if not record[low - 1] else record[low - 1]
            record[low - 1] = f_low
            if not f_low:
                break

            if not record[mid - 1]:
                f_mid = Solve(Graph, source, sink, mid)
                record[mid - 1] = f_mid
                if f_mid:
                    low = mid
                else:
                    high = mid
            else:
                n = low
                break

        if n == 0:
            min_cost = np.inf
        else:
            min_cost = Graph.OptimalCost() / multi_factor
            for arc in range(Graph.NumArcs()):
                if Graph.Tail(arc) != source and Graph.Head(arc) != sink:
                    if Graph.Flow(arc) > 0:
                        row_idx.append(Graph.Tail(arc) - 1)
                        col_idx.append(Graph.Head(arc) - max(start_sets) - 1)

    sz = cost.shape
    unmatched_rows = np.array(list(set(range(sz[0])) - set(row_idx)))
    unmatched_cols = np.array(list(set(range(sz[1])) - set(col_idx)))

    return row_idx, col_idx, np.sort(unmatched_rows), np.sort(unmatched_cols), min_cost


def BuildGraphFromDense(costs, threshold = None):
    """Build Flow from cost matrix

    Parameters
    ----------
    costs: ndarray
        costs between each track_ids with dection_ids
    threshold: float
        if cost > threshold, then will not be considered

    Returns
    ----------
    start_sets: set
        node ids for tracklets except for entry node
    end_sets: set
        node ids for detections except for exit node
    Graph: Object
        Graph for MinCostFlow Solver

    """
    cost_c = deepcopy(np.atleast_2d(costs))

    if threshold is None:
        threshold = inf_cost
    start_nodes, end_nodes = np.where(cost_c < threshold)


    # initialization
    start_sets = set(start_nodes + 1)
    end_sets = set(end_nodes + max(start_sets) + 1)
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # set entry nodes
    for node in start_sets:
        min_cost_flow.AddArcWithCapacityAndUnitCost(0, int(node), 1, 0)

    # set transition nodes
    for i in range(len(start_nodes)):
        start_node = start_nodes[i] + 1
        end_node = end_nodes[i] + max(start_sets) + 1
        min_cost_flow.AddArcWithCapacityAndUnitCost(int(start_node), int(end_node), 1, int(multi_factor*cost_c[start_nodes[i]][end_nodes[i]]))

    #total_sets = start_sets | end_sets

    # set flow numbers
    n = min(len(start_sets), len(end_sets))
    #supplies = [n] + [0]*len(total_sets) + [-n]
    source = 0
    sink = max(end_sets) + 1
    #tasks = len(end_sets)
    #total_sets.add(0)
    #total_sets.add(sink)

    # set exit nodes
    for node in end_sets:
        min_cost_flow.AddArcWithCapacityAndUnitCost(int(node), int(sink), 1, 0)

    # Add node supplies.
    #total_list = list(total_sets)
    #total_list.sort()
    #min_cost_flow.SetNodeSupply(int(source), n)
    #min_cost_flow.SetNodeSupply(int(sink), -n)
    #for i in range(len(supplies)):
    #    min_cost_flow.SetNodeSupply(int(total_list[i]), int(supplies[i]))

    return start_sets, end_sets, min_cost_flow






