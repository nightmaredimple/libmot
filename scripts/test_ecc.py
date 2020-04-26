# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 4/11/2019

import numpy as np
import cv2
import matplotlib.pyplot as plt
from libmot.motion import ECC, AffinePoints


# parameter
thresh = 0.7                      # threshold for detection
war_mode = cv2.MOTION_EUCLIDEAN                # numbers of matched points to be considered
eps = 1e-2                    # levels for orb
max_iter = 100               # numbers of features to be extract
scale = 0.1                # number of matched points to be drawn
src_index = 502               # src image index
dst_index = 505               # dst image index
dir_path = 'E:\\datasets\\MOT17\\train\\MOT17-13-SDP\\'


# prefetch
src = cv2.imread(dir_path + 'img1\\00%04d.jpg' % src_index)
dst = cv2.imread(dir_path + 'img1\\00%04d.jpg' % dst_index)

dets = np.genfromtxt(dir_path + 'det\\det.txt', delimiter=',')
dets = dets[(dets[:, 0] == src_index) & (dets[:, 6] > thresh), 2:6]
dets = dets.astype(np.int32)


# get warp matrix by ecc,
# if you want to speed up, you can make eps↑, max_iter↓， scale↓
# if you choose align, then will get aligned src
warp_matrix, src_aligned = ECC(src, dst, warp_mode=war_mode, eps=eps,
                               max_iter=max_iter, scale=scale, align=True)

print('the warp matrix is:')
print(warp_matrix)

# affine points from src to aligned src by warp matrix
n = len(dets)
points = np.c_[dets[:, :2], dets[:, :2] + dets[:, 2:] - 1]
points_aligned = AffinePoints(points.reshape(n * 2, 2), warp_matrix)
points_aligned = points_aligned.reshape(n, 4)
boxes_aligned = np.c_[points_aligned[:, :2], points_aligned[:, 2:] - points_aligned[:, :2] + 1]

# visualization
draw = np.concatenate((src, dst), axis=1)
sz = src.shape
for (bbox, aligned_bbox) in zip(dets, boxes_aligned):
    x_tl = (bbox[0], bbox[1])
    x_br = (bbox[0] + bbox[2], bbox[1] + bbox[3])
    x_center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
    y_tl = (aligned_bbox[0] + sz[1], aligned_bbox[1])
    y_br = (aligned_bbox[0] + aligned_bbox[2] + sz[1], aligned_bbox[1] + aligned_bbox[3])
    y_center = (int(aligned_bbox[0] + aligned_bbox[2] / 2 + sz[1]), int(aligned_bbox[1] + aligned_bbox[3] / 2))
    cv2.rectangle(draw, x_tl, x_br, (0, 255, 255), 5)
    cv2.rectangle(draw, y_tl, y_br, (0, 0, 155), 5)
    cv2.line(draw, x_center, y_center, (0, 255, 0), 3)
plt.imshow(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
plt.title('ECC Aligned Boxes')
plt.show()







