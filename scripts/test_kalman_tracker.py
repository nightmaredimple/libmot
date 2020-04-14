# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 7/11/2019

import numpy as np
import cv2
import matplotlib.pyplot as plt
from libmot.tracker import LinearMotion,chi2inv95
from copy import deepcopy
from libmot.utils import iou_blocking, evaluation_mot
from libmot.data_association import LinearAssignment, MCFAssignment
import colorsys

# parameter
track_len = 654               # total tracking length
thresh_det = 0.75             # threshold for detection
thresh_track = chi2inv95[4]   # threshold for data association, specifically the mahalanobis distance
fading_memory = 1.14          # fading memory for prediction
dt = 0.15                     # time step for prediction
std_weight_position = 0.04    # std of position prediction
std_weight_velocity = 0.05    # std of velocity prediction
patience = 2                  # patience for waiting reconnection
min_len = 4                   # mininum length of active trajectory
dir_path =  'E:\\datasets\\MOT17\\train\\MOT17-10-SDP\\'

# prefetch
gt = np.genfromtxt(dir_path + 'gt\\gt.txt', delimiter = ',')
gt = gt[(gt[:, 0] <= track_len)&(gt[:, 6] == 1) , :]
mask = (gt[:, 7] == 1) | (gt[:, 7] == 2) | (gt[:, 7] == 7)
gt = gt[mask].astype(np.int32)

dets = np.genfromtxt(dir_path + 'det\\det.txt', delimiter = ',')
dets = dets[(dets[:, 0] <= track_len)&(dets[:, 6] > thresh_det) , :]
dets = dets.astype(np.int32)

tracks = []
record = []

# begin track
total_id = len(dets[dets[:, 0] == 1])
for i, det in enumerate(dets[dets[:, 0] == 1]):
    tracks.append({'id' : i + 1, 'pause': 0,
                   'kf': LinearMotion(det[2:6], fading_memory = fading_memory, \
                                      dt = dt, std_weight_position = std_weight_position, \
                                      std_weight_velocity = std_weight_velocity)
                   })
    record.append([1, i+1, det[2], det[3], det[4], det[5], 1])

for i in range(2, track_len+1):
    det = dets[dets[:, 0] == i]
    cost = (thresh_track + 1)*np.ones((len(tracks), len(det)))
    save = [None] * len(tracks)

    track_copy = deepcopy(tracks)
    track_boxes = np.zeros((len(tracks), 4))

    # predict
    for j, track in enumerate(tracks):
        track['kf'].predict()
        track_boxes[j] = track['kf'].x

    # iou_blocking
    if len(tracks) > 0 and len(det) > 0:
        keep = iou_blocking(track_boxes, det[:, 2:6], 2*track_boxes[:, 2:])
        xs = np.zeros((det.shape[0], tracks[0]['kf'].x_dim, 1))
        Ps = np.zeros((det.shape[0], tracks[0]['kf'].x_dim, tracks[0]['kf'].x_dim))
        ds = np.zeros(det.shape[0])

        # update
        for j, track in enumerate(tracks):
            xs[keep[j], :, :], Ps[keep[j], :, :], ds[keep[j]] = track['kf'].batch_filter(det[keep[j], 2:6])
            save[j] = {'xs': deepcopy(xs), 'Ps': deepcopy(Ps)}
            cost[j, keep[j]] = ds[keep[j]]
        # data association
        row_idx, col_idx, unmatched_rows, unmatched_cols, _ = LinearAssignment(cost, threshold=thresh_track, method = 'KM')
        #row_idx, col_idx, unmatched_rows, unmatched_cols, _ = MCFAssignment(cost, threshold=thresh_track)
    else:
        row_idx = []
        col_idx = []
        unmatched_rows = np.arange(len(tracks))
        unmatched_cols = np.arange(len(det))

    for r, c in zip(row_idx, col_idx):

        tracks[r]['kf'].kf.x = save[r]['xs'][c, :, :]
        tracks[r]['kf'].kf.P = save[r]['Ps'][c, :, :]
        tracks[r]['pause'] = 0

    for r in np.flip(unmatched_rows, 0):
        if tracks[r]['pause'] >= patience:
            del tracks[r]
        else:
            tracks[r]= deepcopy(track_copy[r])
            tracks[r]['kf'].predict()
            tracks[r]['pause'] += 1

    for c in unmatched_cols:
        tracks.append({'id': total_id + 1, 'pause': 0,
                       'kf': LinearMotion(det[c, 2:6], fading_memory=fading_memory, \
                                          dt=dt, std_weight_position=std_weight_position, \
                                          std_weight_velocity=std_weight_velocity)
                       })
        total_id += 1

    for track in tracks:
        if track['pause'] == 0:
            record.append([i, track['id'], track['kf'].x[0], track['kf'].x[1],
                           track['kf'].x[2], track['kf'].x[3], 1])
        else:
            record.append([i, track['id'], track['kf'].x[0], track['kf'].x[1],
                           track['kf'].x[2], track['kf'].x[3], 0])

record = np.array(record)

# post processure
max_id = record[:, 1].flatten().max()
new_record = None
for i in range(1, max_id + 1):
    temp = record[record[:, 1] == i]
    index = int(temp[:, -1].nonzero()[0][-1])
    temp = temp[:(index+1), :-1]
    if len(temp) > min_len or temp[-1, 0] == track_len or temp[0, 0] > track_len - min_len - 1:
        if new_record is not None:
            new_record = np.r_[new_record, temp]
        else:
            new_record = temp

print(evaluation_mot(gt, new_record))

# visualization(1->15frame)

rgb = lambda x: colorsys.hsv_to_rgb((x * 0.41) % 1, 1., 1. - (int(x * 0.41) % 4) / 5.)
colors = lambda x: (int(rgb(x)[0]*255),int(rgb(x)[1]*255),int(rgb(x)[2]*255))

src = cv2.imread(dir_path + 'img1\\000001.jpg')
dst = cv2.imread(dir_path + 'img1\\000015.jpg')
draw = np.concatenate((src,dst), axis=1)
sz = src.shape
boxes = new_record[new_record[:, 0] == 1]
id_list = list(boxes[:, 1])
boxes = boxes[:, 2:6]

track_boxes = []

for i in id_list:
    t = new_record[(new_record[:, 0] == 15) & (new_record[:, 1] == i)]
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
plt.title("kalman tracker")
plt.show()
