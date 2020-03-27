# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 4/11/2019

import cv2
import numpy as np


def drawlines(img1, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''

    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 2)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
    return img1


class Epipolar(object):
    def __init__(self, feature_method = 'orb', match_method = 'brute force',
                 metric = cv2.NORM_HAMMING, n_points = 50, nfeatures = 500,
                 scaleFactor = 1.2, nlevels = 8):
        """Using Epipolar Geometry to Estimate Camara Motion

        Parameters
        ----------
        feature_method : str
            the method of feature extraction, the default is ORB, more methods will be added in the future
        match_method : str
            the method of feature matching, the default is brute force, more methods will be added in the future
        metric: metrics in cv2
            distance metric for feature matching
        n_points: int
            numbers of matched points to be considered
        nfeatures: int
            numbers of features to be extract
        scaleFactor: float
            scale factor for orb
        nlevels: float
            levels for orb
        """
        self.metric = metric
        if feature_method == 'orb':
            self.feature_extractor = cv2.ORB_create(nfeatures = nfeatures,
                                                    scaleFactor = scaleFactor, nlevels = nlevels)
        if match_method == 'brute force':
            self.matcher = cv2.BFMatcher(metric, crossCheck=True)

        self.n_points = n_points


    def FeatureExtract(self, img):
        """Detect and Compute the input image's keypoints and descriptors

        Parameters
        ----------
        img : ndarray of opencv
            An HxW(x3) matrix of img

        Returns
        -------
        keypoints : List of cv2.KeyPoint
            using keypoint.pt can see (x,y)
        descriptors: List of descriptors[keypoints, features]
            keypoints: keypoints which a descriptor cannot be computed are removed
            features: An Nx32 ndarray of unit8 when using "orb" method
        """
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the keypoints with ORB
        keypoints = self.feature_extractor.detect(img, None)
        # compute the descriptors with ORB
        keypoints, descriptors = self.feature_extractor.compute(img, keypoints)
        return keypoints, descriptors


    def GetFundamentalMat(self, keypoints1, descriptors1, keypoints2, descriptors2):
        """Estimate FunfamentalMatrix using BF matcher and ransac
            [p2;1]^T K^(-T) E K^(-1) [p1;1] = 0, T means transpose, K means the intrinsic matrix of camera
            F = K^(-T) E K^(-1)

        Parameters
        ----------
        keypoints : List of cv2.KeyPoint
            using keypoint.pt can see (x,y)
        descriptor : ndarray
            An Nx32 matrix of descriptors

        Returns
        -------
        F: ndarray
            A 3x3 Matrix of Fundamental Matrix
        mask: ndarray
            A Nx1 Matrix of those inline points
        pts1: List of cv2.KeyPoint
            keypoints matched
        pts2: List of cv2.KeyPoint
            keypoints matched
        matches : List of matches
            distance - distance of two points,
            queryIdx - query image's descriptor id, default is the second image
            trainIdx - train image's descriptor id, default is the second image
            imageIdx - train image's id, default is 0
        """
        # matching points
        matches = self.matcher.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        pts1 = []
        pts2 = []
        for i, match in enumerate(matches):
            if i >= self.n_points:
                break
            pts1.append(keypoints1[match.queryIdx].pt)
            pts2.append(keypoints2[match.trainIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        matches = matches[:self.n_points]

        ## Estimate Fundamental Matrix by ransac, distance_threshold = 1, confidence_threshold = 0.99
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1, 0.99)

        return F, mask, pts1, pts2, matches


    def EstimateBox(self, boxes, F):
        """Estimate box in target image by Fundamental Matrix

        Parameters
        ----------
        boxes : array like
            A Nx4 matrix of boxes in source images (x,y,w,h)
        F : ndarray
            A 3x3 Fundamental Matrix

        Returns
        -------
        aligned_boxes: ndarray
            A Nx4 matrix of boxes in source images (x,y,w,h)

        Method
        -------
            L = ||Bi^T F Ai||2 + ||(A2-A0)+(B2-B0)||2
            A is the four corner of box in source image
            B is the four corner of aligned box in target image
            A0,B0:top left corner of box, [x;y;1]
            A1,B1:top right corner of box
            A2,B2:bottom left corner of box
            A3,B3:bottom right corner of box
            the height and width of boxes and aligned boxes are assumed to be same
            we can use greedy strategy: make M = A^T F^T
            then:
                M11   x1   +   M12  y1   + M13 = 0
                M21 (x1+w) +   M22  y1   + M23 = 0
                M31   x1   +   M32 y1+h  + M33 = 0
                M41 (x1+w) +  M42 (y1+h) + M43 = 0
            =>
                M[:2][x;y] + M[:3]+[0;M21w;M32h;M41w+M42h] = 0 ->Ax = b
                x = (pseudo inverse of A )b

        """
        boxes = np.asarray(boxes)
        if boxes.ndim == 1:
            boxes = boxes[np.newaxis, :]
        aligned_boxes = np.zeros(boxes.shape)

        for i, bbox in enumerate(boxes):
            w = bbox[2]
            h = bbox[3]
            AT = np.array([[bbox[0]   , bbox[1]    , 1],
                          [bbox[0] + w, bbox[1]    , 1],
                          [bbox[0]    , bbox[1] + h, 1],
                          [bbox[0] + w, bbox[1] + h, 1]])
            M = AT @ F.T
            b = -M[:, 2] - np.array([0, M[1][0]*w, M[2][1]*h, M[3][0]*w+M[3][1]*h])
            aligned_tl = np.linalg.pinv(M[:,:2]) @ b

            aligned_boxes[i, 0] = aligned_tl[0]
            aligned_boxes[i, 1] = aligned_tl[1]
            aligned_boxes[i, 2] = w
            aligned_boxes[i, 3] = h

        return aligned_boxes.astype(np.int32)


    def DrawMatches(self, src, dst, keypoints1, keypoints2, matches, n = None):
        """Draw matches between source image with target image

        Parameters
        ----------
        src : ndarray
            A HxW matrix of opencv image
        dst : ndarray
            A HxW matrix of opencv image
        keypoints1: ndarray
            A Nx2 matrix of keypoints in src image
        keypoints2: ndarray
            A Nx2 matrix of keypoints in dst image
        matches: List of matches
        n: int
            numbers of matches to be drawn
        Returns
        -------
        draw: ndarray
            A Hx2W matrix of opencv image
        """
        assert src.shape == dst.shape, "source image must be the same format with target image"
        if n is None:
            n = self.n_points
        n = min(n, len(matches))
        draw = cv2.drawMatches(src, keypoints1, dst, keypoints2, matches[: n], None, flags = 2)
        return draw


    def DrawAlignedBox(self, src, dst, boxes, aligned_boxes):
        """Draw matches boxes between source image with target image

        Parameters
        ----------
        src : ndarray
            A HxW(x3) matrix of opencv image
        dst : ndarray
            A HxW(x3) matrix of opencv image
        boxes: array like
            A Nx4 matrix of boxes in src image
        aligned_boxes: array like
            A Nx4matrix of aligned boxes in dst image
        Returns
        -------
        draw: ndarray
            A Hx2W matrix of opencv image
        """
        assert src.shape == dst.shape, "source image must be the same format with target image"

        boxes, aligned_boxes = np.asarray(boxes), np.asarray(aligned_boxes)
        sz = src.shape
        draw = np.concatenate((src, dst), axis=1)

        for (bbox, aligned_bbox) in zip(boxes, aligned_boxes):
            x_tl = (bbox[0], bbox[1])
            x_br = (bbox[0] + bbox[2], bbox[1] + bbox[3])
            x_center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
            y_tl = (aligned_bbox[0] + sz[1], aligned_bbox[1])
            y_br = (aligned_bbox[0] + aligned_bbox[2] + sz[1], aligned_bbox[1] + aligned_bbox[3])
            y_center = (int(aligned_bbox[0] + aligned_bbox[2] / 2 + sz[1]), int(aligned_bbox[1] + aligned_bbox[3] / 2))
            cv2.rectangle(draw, x_tl, x_br, (0, 255, 255), 5)
            cv2.rectangle(draw, y_tl, y_br, (0, 0, 155), 5)
            cv2.line(draw, x_center, y_center, (0, 255, 0), 3)

        return draw


    def DrawCorrespondEpilines(self, src, dst, pts1, pts2, F):
        """Draw Correspond Epilines on the image by Fundamental matrix

        Parameters
        ----------
        src : ndarray
            A HxW(x3) matrix of opencv image
        dst : ndarray
            A HxW(x3) matrix of opencv image
        pts1: List of cv2.KeyPoint
            keypoints matched of src image
        pts1: List of cv2.KeyPoint
            keypoints matched of dst image
        F: ndarray
            A 3x3 Matrix of Fundamental Matrix

        Returns
        -------
        draw: ndarray
            A HxW matrix of opencv image
        """
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img1 = drawlines(src, lines1, pts1, pts2)

        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img2 = drawlines(dst, lines2, pts2, pts1)
        return img1, img2






