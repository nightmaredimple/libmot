# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 4/11/2019

import numpy as np
import cv2
import matplotlib.pyplot as plt
from libmot.motion import Epipolar

# parameter
thresh = 0.7                  # threshold for detection
n_points = 100                # numbers of matched points to be considered
n_levels = 6                  # levels for orb
n_features = 50               # numbers of features to be extract
n_matches = 30                # number of matched points to be drawn
src_index = 498               # src image index
dst_index = 500               # dst image index
dir_path = 'E:\\datasets\\MOT17\\train\\MOT17-13-SDP\\'

# prefetch
src = cv2.imread(dir_path + 'img1\\00%04d.jpg' % src_index)
dst = cv2.imread(dir_path + 'img1\\00%04d.jpg' % dst_index)

dets = np.genfromtxt(dir_path + 'det\\det.txt', delimiter=',')
dets = dets[(dets[:, 0] == src_index) & (dets[:, 6] > thresh), 2:6]
dets = dets.astype(np.int32)
import time

# init Epipolar, n_points means thenumbers of matched points to be considered
Model = Epipolar(n_points=n_points, nlevels=n_levels, nfeatures=n_features)

# Feature Extract
keypoints1, descriptors1 = Model.FeatureExtract(src)
keypoints2, descriptors2 = Model.FeatureExtract(dst)

# GetFundamentalMat
F, mask, pts1, pts2, matches = Model.GetFundamentalMat(keypoints1, descriptors1, keypoints2, descriptors2)
print('The Fundamental matrix is:')
print(F)

# Estimate Box by F
aligned_box = Model.EstimateBox(dets, F)

# visualization,n is the number of matches to be drawn, default is n_points
draw1 = Model.DrawMatches(src, dst, keypoints1, keypoints2, matches, n=n_matches)
draw2 = Model.DrawAlignedBox(src, dst, dets, aligned_box)
img1, img2 = Model.DrawCorrespondEpilines(src, dst, pts1, pts2, F)


plt.subplot(221)
plt.imshow(cv2.cvtColor(draw1,cv2.COLOR_BGR2RGB))
plt.title('Matched Points')
plt.subplot(222)
plt.imshow(cv2.cvtColor(draw2,cv2.COLOR_BGR2RGB))
plt.title('Aligned Boxes')
plt.subplot(223)
plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
plt.title('Src Epilines')
plt.subplot(224)
plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
plt.title('Dst Epilines')
plt.show()
