# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 1/12/2019

import numpy as np
import cv2
import matplotlib.pyplot as plt

from libmot.data_association  import MinCostFlowTracker
from copy import deepcopy
import time
from libmot.utils import iou
import motmetrics as mm
import pandas as pd
import colorsys

# parameters
default = {'entry_exit_cost': 5, 'thresh': 1.8,
           'miss_rate': 0.8, 'duration': 12}
# prefetch
images = []
track_len = 655
for i in range(1, track_len):
    images.append(cv2.imread('E:\\datasets\\MOT17\\train\\MOT17-10-SDP\\img1\\00%04d.jpg' % i))

gt = np.genfromtxt('E:\\datasets\\MOT17\\train\\MOT17-10-SDP\\gt\\gt.txt', delimiter=',')
gt = gt[(gt[:, 0] < track_len) & (gt[:, 6] == 1), :]
mask = (gt[:, 7] == 1) | (gt[:, 7] == 2) | (gt[:, 7] == 7)
gt = gt[mask].astype(np.int32)

dets = np.genfromtxt('E:\\datasets\\MOT17\\train\\MOT17-10-SDP\\det\\det.txt', delimiter=',')
dets = dets[(dets[:, 0] < track_len) & (dets[:, 6] > 0.7), :]

dets = dets.astype(np.int32)


# functions
def observation_model(**kwargs):
    """Compute Observation Cost

    Parameters
    ------------
    scores: array like
        (N,) matrix of detection's occlusion ratio

    Returns
    -----------
    costs: ndarray
        (N,) matrix of detection's observation costs
    """
    if 'scores' in kwargs:
        scores = np.array(kwargs['scores']).astype(np.float)
        return np.log(-0.446 * scores + 0.456)
    else:
        return -2


def feature_model(**kwargs):
    """Compute each object's feature

    Parameters
    ------------
    image: ndarray
        bgr image of ndarray
    boxes: array like
        (N,4) matrix of boxes in (x,y,w,h)

    Returns
    ------------
    features: array like
        (N,M,K) matrix of features
    """
    assert 'image' in kwargs and 'boxes' in kwargs, 'Parameters must contail image and boxes'

    boxes = kwargs['boxes']
    image = kwargs['image']
    if len(boxes) == 0:
        return np.zeros((0,))

    boxes = np.atleast_2d(deepcopy(boxes))
    features = np.zeros((boxes.shape[0], 180, 256), dtype=np.float32)

    for i, roi in enumerate(boxes):
        x1 = max(roi[1], 0)
        x2 = max(roi[0], 0)
        x3 = max(x1 + 1, x1 + roi[3])
        x4 = max(x2 + 1, x2 + roi[2])
        cropped = image[x1:x3, x2:x4, :]
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        features[i] = deepcopy(hist)

    return features


def transition_model(**kwargs):
    """Compute costs between track and detection

    Parameters
    ------------
    miss_rate: float
        the similarity for track and detection will be multiplied by miss_rate^(time_gap - 1)
    time_gap: int
        number of frames between track and detection
    predecessor_boxes: ndarray
        (N,4) matrix of boxes in track's frame
    boxes: ndarray
        (M,4) matrix of boxes in detection's frame
    predecessor_features: ndarray
        (N,180, 256) matrix of features in track's frame
    features: ndarray
        (M,180, 256) matrix of features in detection's frame
    frame_idx: int
        frame id, begin from 1

    Returns
    ------------
    costs: ndarray
        (N,M) matrix of costs between track and detection
    """
    miss_rate = kwargs['miss_rate']
    time_gap = kwargs['time_gap'] - 1
    boxes = kwargs['boxes']
    predecessor_boxes = kwargs['predecessor_boxes']
    features = kwargs['features']
    predecessor_features = kwargs['predecessor_features']
    frame_idx = kwargs['frame_idx']

    assert len(boxes) == len(features) and len(predecessor_boxes) == len(predecessor_features), \
        "each boxes and features's length must be same"

    # warp_matrix, _ = ECC(images[frame_idx - time_gap - 2], images[frame_idx - 1],
    #                    warp_mode=cv2.MOTION_EUCLIDEAN,
    #                    eps=0.01, max_iter=100, scale=0.1, align=False)

    if len(boxes) == 0 or len(predecessor_boxes) == 0:
        return np.zeros((0,))
    costs = np.zeros((len(predecessor_boxes), len(boxes)))
    for i, (box1, feature1) in enumerate(zip(predecessor_boxes, predecessor_features)):
        """
        points = np.array([box1[:2],
                           box1[:2] + box1[2:] - 1])
        points_aligned = AffinePoints(points.reshape(2, 2), warp_matrix)
        points_aligned = points_aligned.reshape(1, 4)
        boxes_aligned = np.c_[points_aligned[:, :2],
                              points_aligned[:, 2:] - points_aligned[:, :2] + 1]
        boxes_aligned[:, 2:] = np.clip(boxes_aligned[:, 2:], 1, np.inf)
        box1 = boxes_aligned.squeeze()
        """
        for j, (box2, feature2) in enumerate(zip(boxes, features)):
            if max(abs(box1[0] - box2[0]), abs(box1[1] - box2[1])) > max(box1[2], box1[3], box2[2], box2[3]):
                costs[i, j] = np.inf
            else:
                costs[i, j] = -np.log(0.5 * float(iou(box1, box2)) * (miss_rate ** time_gap) \
                                      + 0.5 * (1 - cv2.compareHist(feature1, feature2,
                                                                   cv2.HISTCMP_BHATTACHARYYA)) + 1e-5)

    return costs


def get_tracks(entry_exit_cost=None, thresh=None,
               miss_rate=None, duration=None):
    entry_exit_cost = default['entry_exit_cost'] if entry_exit_cost is None else entry_exit_cost
    thresh = default['thresh'] if thresh is None else thresh
    miss_rate = default['miss_rate'] if miss_rate is None else miss_rate
    duration = default['duration'] if duration is None else duration

    record = []
    track_model = MinCostFlowTracker(observation_model=observation_model,
                                     transition_model=transition_model, feature_model=feature_model,
                                     entry_exit_cost=entry_exit_cost, min_flow=20,
                                     max_flow=1000, miss_rate=miss_rate, max_num_misses=duration,
                                     cost_threshold=thresh)

    for i in range(1, track_len):
        det = dets[(dets[:, 0] == i), :]
        track_model.process(boxes=det[:, 2: 6].astype(np.int32),
                            scores=det[:, 6], image=images[i - 1],
                            frame_idx=i)

    trajectory = track_model.compute_trajectories()
    for i, t in enumerate(trajectory):
        for j, box in enumerate(t):
            record.append([box[0] + 1, i + 1, box[2][0], box[2][1], box[2][2], box[2][3]])
    record = np.array(record)
    record = record[np.argsort(record[:, 0])]

    return record


def evaluation(tracks):
    gts = pd.DataFrame(gt[:, :6], columns=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
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


# test
default = {'entry_exit_cost': 1, 'thresh': 1,
           'miss_rate': 0.8, 'duration': 2}

time0 = time.time()
track = np.array(get_tracks())
print(time.time() - time0)
print(evaluation(track))

rgb = lambda x: colorsys.hsv_to_rgb((x * 0.41) % 1, 1., 1. - (int(x * 0.41) % 4) / 5.)
colors = lambda x: (int(rgb(x)[0] * 255), int(rgb(x)[1] * 255), int(rgb(x)[2] * 255))
draw = np.concatenate((images[0], images[15]), axis=1)
sz = images[0].shape
boxes = track[track[:, 0] == 1]
id_list = list(boxes[:, 1])
boxes = boxes[:, 2:6]

track_boxes = []

for i in id_list:
    t = track[(track[:, 0] == 16) & (track[:, 1] == i)]
    if t.size > 0:
        track_boxes.append(t[:, 2:6].squeeze())
    else:
        track_boxes.append(None)

for i, (bbox, tracked_bbox) in enumerate(zip(boxes, track_boxes)):
    x_tl = (bbox[0], bbox[1])
    x_br = (bbox[0] + bbox[2], bbox[1] + bbox[3])
    x_center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
    cv2.rectangle(draw, x_tl, x_br, colors(i), 5)

    if tracked_bbox is not None:
        y_tl = (tracked_bbox[0] + sz[1], tracked_bbox[1])
        y_br = (tracked_bbox[0] + tracked_bbox[2] + sz[1], tracked_bbox[1] + tracked_bbox[3])
        y_center = (int(tracked_bbox[0] + tracked_bbox[2] / 2 + sz[1]), int(tracked_bbox[1] + tracked_bbox[3] / 2))

        cv2.rectangle(draw, y_tl, y_br, colors(i), 5)
        cv2.line(draw, x_center, y_center, colors(i), 3)

plt.imshow(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
plt.title("offline MinCostFlow Tracker")
plt.show()
