# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 16/11/2019

import numpy as np
from ortools.graph import pywrapgraph
from copy import deepcopy
import time
import torch

inf_cost = 1e+5
multi_factor = 1e+5

# -----------------------------------------------------------------------------------------------------------
# This is new version with better arrangement
class MinCostFlowTracker(object):

    def __init__(self, observation_model = None,
                 transition_model = None, feature_model = None,
                 batch_size = 1, entry_exit_cost = 5, min_flow = 1,
                 max_flow = 50, miss_rate = 0.7, max_num_misses = 10,
                 cost_threshold = -np.log(0.1), powersave = False):
        """Using MinCostFlow to handle data association problem

        Parameters
        --------------
        observation_model: function
            cost for choosing the dection, eg: 0~1 confidence can be cast to ln(1-p) - ln(p)
            note that if the confidence is accepted, the observation cost should be negative
        transition_model: function
            cost for connecting the track with the detection, usually be non-negative,eg:-ln(prob)
        feature_model: function
            get feature for transition models or other models
        batch_size: int
            batch size for feature models
        entry_exit_cost: float
            cost for source -> trajectory and trajectory -> sink
        min_flow: int
            min numbers of trajectories
        max_flow: int
            max numbers of trajectories
        miss_rate: float
            the similarity for track and detection will be multiplied by miss_rate^(time_gap - 1)
        max_num_misses: int
            tracks and detections within max_num_misses can be connected
        cost_threshold: float
            threshold of cost
        powersave: bool
            whether to save history node information
        """
        self.observation_model = observation_model
        self.transition_model = transition_model
        self.feature_model = feature_model
        self.batch_size = batch_size

        self.entry_exit_cost = entry_exit_cost
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.miss_rate = miss_rate
        self.max_num_misses = max_num_misses
        self.cost_threshold = cost_threshold
        self.powersave = powersave

        time_gap_to_probability = np.array([1e-15] + [
            np.power(miss_rate, time_gap - 1)
            for time_gap in range(1,2 + max_num_misses)])
        self.time_gap_to_cost = -np.log(time_gap_to_probability) # each time gap -> cost

        self.trajectories = []   # trajectory in current timestep
        self.current_frame_idx = 0 # current frame id
        self.nodes_in_timestep = [] # List[List[nodes]] within max_num_misses

        self.node = {0: {'type': 'source'}, 1: {'type': 'sink'}} # node record
        self.node_idx = 2 # next_node id
        self.last_frame_id = 2 # last frame's first node

        self.entire_trajectories = []  # total trjectories through entire video

        self.graph = pywrapgraph.SimpleMinCostFlow()

        self.fib = {0: 0, 1: 1}

    def process(self, boxes, scores, image = None, features = None, **kwargs):
        """Process one frame of detections.
        Parameters
        ----------
        boxes : ndarray
            An Nx4 dimensional array of bounding boxes in
            format (top-left-x, top-left-y, width, height).
        scores : ndarray
            An array of N associated detector confidence scores.
        image : Optional[ndarray]
            Optionally, a BGR color image;
        features : Optional[ndarray]
            Optionally, an NxL dimensional array of N feature vectors
            corresponding to the given boxes. If None given, bgr_image must not
            be None and the tracker must be given a feature models for feature
            extraction on construction.
        **kwargs : other parameters that models needed

        Returns
        -------
        trajectories: List[List[Tuple[int, int, ndarray]]]
            Returns [] if the tracker operates in offline mode. Otherwise,
            returns the set of object trajectories at the current time step.
        entire_trajectories: List[List[Tuple[int, int, ndarray]]]
            entire time steps trajectories
        """
        # save the first node id in current frame
        first_node_id = deepcopy(self.node_idx)

        # Compute features if necessary.
        parameters = {'image': image, 'boxes': boxes, 'scores': scores,
                      'miss_rate': self.miss_rate, 'batch_size': self.batch_size}
        parameters.update(kwargs)
        if features is None:
            assert self.feature_model is not None, "No feature models given"
            features = self.feature_model(**parameters)


        # Add nodes to graph for detections observed at this time step.
        observation_costs = (self.observation_model(**parameters)
            if len(scores) > 0 else np.zeros((0,)))
        node_ids = []
        for i, cost in enumerate(observation_costs):
            self.node.update({self.node_idx:
                {
                    "type": 'object',
                    "box": boxes[i],
                    "feature": features[i],
                    "frame_idx": self.current_frame_idx,
                    "box_idx": i,
                    'cost': cost
                }
            } )

            # save object node id to this time step
            node_ids.append(self.node_idx)

            self.node.update({self.node_idx + 1:
                {
                    "type": 'transition',
                }
            })
            self.graph.AddArcWithCapacityAndUnitCost(0, int(self.node_idx), 1, \
                                                     int(multi_factor * self.entry_exit_cost))
            self.graph.AddArcWithCapacityAndUnitCost(int(self.node_idx), int(self.node_idx + 1), \
                                                     1, int(multi_factor * cost))
            self.graph.AddArcWithCapacityAndUnitCost(int(self.node_idx + 1), 1, 1, \
                                                     int(multi_factor * self.entry_exit_cost))
            self.node_idx += 2

        # Link detections to candidate predecessors.
        predecessor_time_slices = (
            self.nodes_in_timestep[-(1 + self.max_num_misses):])
        for k, predecessor_node_ids in enumerate(predecessor_time_slices):
            if len(predecessor_node_ids) == 0 or len(node_ids) == 0:
                continue
            predecessors = [self.node[x] for x in predecessor_node_ids]
            predecessor_boxes = np.asarray(
                [node["box"] for node in predecessors])
            if isinstance(features,np.ndarray):
                predecessor_features = np.asarray(
                    [node["feature"] for node in predecessors])
            else:
                predecessor_features = torch.cat(
                    [node["feature"].unsqueeze(0) for node in predecessors])

            time_gap = len(predecessor_time_slices) - k

            transition_costs = self.transition_model(
                miss_rate = self.miss_rate,
                time_gap = time_gap, predecessor_boxes = predecessor_boxes,
                predecessor_features = predecessor_features,
                boxes = boxes, features = features, **kwargs)

            for i, costs in enumerate(transition_costs):
                for j, cost in enumerate(costs):
                    if cost > self.cost_threshold:
                        continue
                    last_id = int(predecessor_node_ids[i] + 1)
                    self.graph.AddArcWithCapacityAndUnitCost(last_id, int(node_ids[j]), 1,
                                                             int(multi_factor * cost))
        self.nodes_in_timestep.append(node_ids)

        # Compute trajectories

        self.current_frame_idx += 1
        self.last_frame_id = first_node_id

        return self.trajectories, self.entire_trajectories

    def update_latest_node(self, boxes, image = None, **kwargs):
        """update node's box by new boxes

        Parameters
        ------------
        boxes : ndarray
            An Nx4 dimensional array of bounding boxes in
            format (top-left-x, top-left-y, width, height).
        image : Optional[ndarray]
            Optionally, a BGR color image
        """
        parameters = {'boxes': boxes, 'image': image}
        parameters.update(kwargs)

        features = self.feature_model(**parameters)
        for i in range(self.last_frame_id, self.node_idx):
            self.node[i]['box'] = boxes[i - self.last_frame_id]
            self.node[i]['feature'] = features[i - self.last_frame_id]
        for t in self.entire_trajectories:
            if t[-1][0] == self.current_frame_idx - 1:
                t[-1] = (t[-1][0], t[-1][1], deepcopy(boxes[t[-1][1]]))

    def compute_trajectories(self):
        """Compute trajectories over the entire observation sequence

        Returns
        ----------
        List[List[Tuple(int, int, ndarray)]]]
            Returns the set of object trajectories. Each entry contains the
            index of the frame at which the detection occured (see
            next_frame_idx) and the bounding box in
            format (top-left-x, top-left-y, width, height).
        """

        self.trajectories, min_cost = self.fibonacci_search()

        self.entire_trajectories = deepcopy(self.trajectories)

        return self.trajectories

    def binary_search(self, low = 0, high = 500):
        """Use binary search to find the mincostflow

        Parameters
        -----------
        low: int
            lower bound for number of flows
        high: int
            higher bound for number of flows

        Returns
        ----------
        min_cost: int or float
            min cost of flows in graph
        n: int
            number of trajatectories
        """

        low = max(0, low)
        assert low <= high, "lower bound must be lower than high bound"

        n = 0

        if high < 1:
            min_cost = np.inf
        else:
            record = np.array([False] * high)
            while low <= high:
                mid = (low + high) // 2
                f_high = self.solve(high) if not record[high - 1] else record[high - 1]
                record[high - 1] = f_high
                if f_high:
                    n = high
                    break

                f_low = self.solve(low) if not record[low - 1] else record[low - 1]
                record[low - 1] = f_low
                if not f_low:
                    break

                if not record[mid - 1]:
                    f_mid = self.solve(mid)
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
                min_cost = self.graph.OptimalCost() / multi_factor

        return min_cost, n

    def fibonacci(self, n):
        """Use Fibonacci Search to speed up Searching
        there can exist u~v flows(id), so we need to find the min cost flows

        Parameters:
        -----------
        n: int

        Returns:
        -----------
        fn: int
            the n th fibonacci number
        """
        assert n > -1, "n must be non-negative number"

        if n in self.fib:
            return self.fib[n]
        else:
            return self.fib.setdefault(n, self.fibonacci(n - 1) + self.fibonacci(n - 2))

    def fibonacci_search(self):
        """Run Fibonacci Searching to find the min cost flow

        Returns
        -------
        trajectories: List[List]
            List of trajectories
        min_cost: float
            cost of assignments
        """
        k = 0
        r = max(0, self.max_flow - self.min_flow)
        s = self.min_flow
        cost = {}
        trajectories = []

        # find the nearest pos of fibonacci
        while r > self.fibonacci(k):
            k = k + 1

        while k > 1:
            u = min(self.max_flow, s + self.fibonacci(k - 1))
            v = min(self.max_flow, s + self.fibonacci(k - 2))

            if u not in cost:
                self.graph.SetNodeSupply(0, u)
                self.graph.SetNodeSupply(1, -u)
                if self.graph.Solve() == self.graph.OPTIMAL:
                    cost[u] = self.graph.OptimalCost()

                else:
                    cost[u] = np.inf

            if v not in cost:
                self.graph.SetNodeSupply(0, v)
                self.graph.SetNodeSupply(1, -v)
                if self.graph.Solve() == self.graph.OPTIMAL:
                    cost[v] = self.graph.OptimalCost()

                else:
                    cost[v] = np.inf

            if cost[v] == cost[u]:
                s = v
                k = k - 1
            elif cost[v] < cost[u]:
                k = k - 1
            else:
                s = u
                k = k - 2

        self.graph.SetNodeSupply(0, s)
        self.graph.SetNodeSupply(1, -s)

        if self.graph.Solve() == self.graph.OPTIMAL:
            min_cost =  self.graph.OptimalCost() / multi_factor
            hashlist = {0: []}
            # create disjoint set
            for arc in range(self.graph.NumArcs()):
                if self.graph.Flow(arc) > 0:
                    if self.graph.Tail(arc) == 0:
                        hashlist[0].append(self.graph.Head(arc))
                    else:
                        hashlist[self.graph.Tail(arc)] = self.graph.Head(arc)
            for entry in hashlist[0]:
                tracklet = [(
                            self.node[entry]['frame_idx'],
                            self.node[entry]['box_idx'],
                            self.node[entry]['box']
                             )]
                point = hashlist[entry]
                while point != 1:
                    if self.node[point]['type'] == 'object':
                        tracklet.append((
                            self.node[point]['frame_idx'],
                            self.node[point]['box_idx'],
                            self.node[point]['box']
                             ))
                    if point in hashlist:
                        point = hashlist[point]
                    else:
                        break
                trajectories.append(tracklet)

        else:
            min_cost = inf_cost

        return trajectories, min_cost

    def get_trajectory(self):
        """Get trajectory from graph

        Returns
        -----------
        trajectory: List[Tuple(int, int, ndarray)]
            trajectory in one timestep List[(frame_id, box_id, box)]

        """
        self.trajectories = []
        for arc in range(self.graph.NumArcs()):
            if self.graph.Tail(arc) != 0 and self.graph.Head(arc) != 1:
                if self.graph.Flow(arc) > 0:
                    if self.node[self.graph.Head(arc)]['type'] == 'object':
                        self.trajectories.append([
                            (
                                self.node[self.graph.Tail(arc)]['frame_idx'],
                                self.node[self.graph.Tail(arc)]['box_idx'],
                                self.node[self.graph.Tail(arc)]['box']
                             ),
                            (
                                self.node[self.graph.Head(arc)]['frame_idx'],
                                self.node[self.graph.Head(arc)]['box_idx'],
                                self.node[self.graph.Head(arc)]['box']
                            )
                            ]
                        )
            elif self.graph.Head(arc) == 1 and self.graph.Flow(arc) == 0:
                self.trajectories.append([
                    (
                        self.node[self.graph.Tail(arc)]['frame_idx'],
                        self.node[self.graph.Tail(arc)]['box_idx'],
                        self.node[self.graph.Tail(arc)]['box']
                    )
                ])
            '''
            elif self.graph.Tail(arc) == 0 and self.graph.Flow(arc) == 0:
                self.trajectories.append([
                    (
                        self.node[self.graph.Head(arc)]['frame_idx'],
                        self.node[self.graph.Head(arc)]['box_idx'],
                        self.node[self.graph.Head(arc)]['box']
                    )
                ])
            '''

        return self.trajectories

    def node2trajectory(self, index1, index2):
        """Convert nodes [index1, index2) to trjectories

        Parameters
        -----------
        index1: int
            lower bound of node id
        index2: int
            higher bound of node id

        Returns
        -----------
        trajectory: List[Tuple(int, int, ndarray)]
            trajectory in one timestep List[(frame_id, box_id, box)]
        """

        assert index1 < index2, 'index1 must be lower than index2'

        assert max(self.node.keys()) >= index2 - 1, 'index2 must be equal lower than max node id+1'

        self.trajectories = []
        for i in range(index1, index2):
            self.trajectories.append([
                (
                    self.node[i]['frame_idx'],
                    self.node[i]['box_idx'],
                    self.node[i]['box']
                ),
            ])

        return self.trajectories

    def merge_trajectories(self, src, dst):
        """ Merge trajectories between src and dst :src+dst -> dst

        Parameters
        -------------
        src: List[List[Tuple(int, int, ndarray)]]
            source trajectory
        dst: List[List[Tuple(int, int, ndarray)]]
            target trajectory

        Returns
        -----------
        dst: List[List[Tuple(int, int, ndarray)]]
            Merged trajectories
        """
        if len(dst) == 0:
            return src

        if len(src) == 0:
            return dst

        last_node_dict = {(trajectory[-1][0], trajectory[-1][1]) : i  \
                          for i, trajectory in enumerate(dst)}
        first_node_dict = {(trajectory[0][0], trajectory[0][1]) : j  \
                           for j, trajectory in enumerate(src)}

        merged_id = []
        for key, value in last_node_dict.items():
            if key in first_node_dict.keys():
                merged_id.append(first_node_dict[key])
                if len(src[first_node_dict[key]]) > 1:
                    dst[value].extend(deepcopy(src[first_node_dict[key]][1:]))
        remained = [t for i,t in enumerate(src) if i not in merged_id]
        dst = dst + remained
        return dst

    def solve(self, n):
        """Solve MinCostFlow with n flows

        Parameters:
        ------------
        n: int
            numbers of flows

        Returns:
        ------------
        isOptimal: bool
            whether the Graph is feasible
        """
        self.graph.SetNodeSupply(0, n)
        self.graph.SetNodeSupply(1, -n)

        return self.graph.Solve() == self.graph.OPTIMAL
