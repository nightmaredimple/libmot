# -*- coding: utf-8 -*-
# Author : Huang Piao
# Email  : huangpiao2985@163.com
# Date   : 6/11/2019

from __future__ import division
import numpy as np
from libmot.motion import KalmanFilter
from copy import deepcopy

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9).
used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class LinearMotion(object):
    """MOT Tracker using Kalman Filter
       refer to https://github.com/nwojke/deep_sort
    """

    def __init__(self, box, fading_memory = 1.0, dt = 1.0, std_weight_position = 0.05, std_weight_velocity = 0.00625):
        """Tracking bounding boxes in assumption of uniform linear motion

        The 8-dimensional state space

            x, y, a, h, vx, vy, va, vh

        contains the bounding box center position (x, y), aspect ratio a, height h,
        and their respective velocities.

        Object motion follows a constant velocity models. The bounding box location
        (x, y, a, h) is taken as direct observation of the state space (linear
        observation models).

        Parameters
        --------------
        box: array like
            1x4 matrix of boxes (x,y,w,h)
        fading_memory: float
            larger means fading more
        dt: float
            time step for each update
        std_weight_position: float
            std for position
        std_weight_velocity:flaot
            std for velovity

        """
        box = np.atleast_2d(box) #1x4

        # initialization
        self.x_dim = 8
        self.z_dim = 4
        self.dt = dt

        state = self.box2state(box)
        state = np.r_[state, np.zeros_like(state)] #8x1

        self.F = np.eye(self.x_dim, self.x_dim)
        for i in range(self.z_dim):
            self.F[i, self.z_dim + i] = self.dt

        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

        std = [
            2 * self._std_weight_position * state[3][0],
            2 * self._std_weight_position * state[3][0],
            1e-2,
            2 * self._std_weight_position * state[3][0],
            10 * self._std_weight_velocity * state[3][0],
            10 * self._std_weight_velocity * state[3][0],
            1e-5,
            10 * self._std_weight_velocity * state[3][0]]
        covariance = np.diag(np.square(std))

        self.H = np.eye(self.z_dim, self.x_dim)

        self.kf = KalmanFilter(dim_x = self.x_dim, dim_z = self.z_dim , x = state,
                               P = covariance, F = self.F, H = self.H)
        self.kf.alpha = fading_memory
        self._x = self.kf.x
        self._mahalanobis = self.kf.mahalanobis

    def predict(self):
        """Predict next state (prior) using the Kalman filter state propagation equations:
                             x = Fx + Bu
                             P = fading_memory*FPF^T + Q
        """
        std_pos = [
            self._std_weight_position * self.kf.x[3][0],
            self._std_weight_position * self.kf.x[3][0],
            1e-2,
            self._std_weight_position * self.kf.x[3][0]]
        std_vel = [
            self._std_weight_velocity * self.kf.x[3][0],
            self._std_weight_velocity * self.kf.x[3][0],
            1e-5,
            self._std_weight_velocity * self.kf.x[3][0]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        self.kf.predict(Q = motion_cov)

    def update(self, measurement):
        """
        Update Process, add a new measurement (z) to the Kalman filter.
                    K = PH^T(HPH^T + R)^-1
                    y = z - Hx
                    x = x + Ky
                    P = (1 - KH)P or P = (I-KH)P(I-KH)' + KRK'

        Parameters
        --------------
        measurement: array like
            1x4 matrix of boxes (x,y,w,h)

        """
        box = np.atleast_2d(measurement)  # 1x4
        z = self.box2state(box)   # 4x1

        std = [
            self._std_weight_position * self.kf.x[3][0],
            self._std_weight_position * self.kf.x[3][0],
            1e-1,
            self._std_weight_position * self.kf.x[3][0]]
        innovation_cov = np.diag(np.square(std))

        self.kf.update(z = z, R = innovation_cov)

    def batch_filter(self, zs):
        """ Batch processes a sequences of measurements.

        Parameters
        ----------

        zs : list-like
            list of measurements at each time step `self.dt`. Missing
            measurements must be represented by `None`.

        Returns
        -------

        means : np.array((n,dim_x,1))
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.

        covariance : np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        mahalanobis: np.array((n,1))
            array of the mahalanobises for each time step during the update

        """
        zs = np.atleast_2d(zs)
        n = zs.shape[0]

        # mean estimates from Kalman Filter
        x_copy = deepcopy(self.kf.x)
        means = np.zeros((n, self.x_dim, 1))

        # state covariances from Kalman Filter
        covariances = np.zeros((n, self.x_dim, self.x_dim))
        mahalanobis = np.zeros(n)

        std = [
            self._std_weight_position * self.kf.x[3][0],
            self._std_weight_position * self.kf.x[3][0],
            1e-1,
            self._std_weight_position * self.kf.x[3][0]]
        innovation_cov = np.diag(np.square(std))

        if n > 0:
            for i, z in enumerate(zs):
                self.kf.x = deepcopy(x_copy)

                measurement = self.box2state(z)  # 4x1
                self.kf.update(z = measurement, R = innovation_cov)
                means[i, :] = deepcopy(self.kf.x)
                if i == 0:
                    covariances = np.tile(self.kf.P[np.newaxis, :, :],(n,1,1))

                mahalanobis[i] = deepcopy(self.kf._mahalanobis)

        return (means, covariances, mahalanobis)

    def box2state(self, box):
        """ Convert box(x,y,w,h) to state(x,y,a,h)
        Parameters
        --------------
        box: array like
            1x4 matrix of boxes (x,y,w,h)

        Returns
        --------------
        state: ndarray
            4x1 matrix of boxes (cx,cy,a,h)
        """
        state = np.atleast_2d(deepcopy(box))
        state = state.astype(np.float)
        state[:, :2] = state[:, :2] + state[:, 2:] / 2 - 1
        state[:, 2] = state[:, 2] / state[:, 3]

        return state.T

    def state2box(self, state):
        """ Convert  state(cx,cy,a,h) to box(x,y,w,h)
        Parameters
        --------------
        state: array like
            8x1 matrix of boxes (cx,cy,a,h)

        Returns
        --------------
        box: ndarray
            1x4  matrix of boxes (x,y,w,h)
        """

        box = deepcopy(state.squeeze()[:4])
        box[2] = box[2] * box[3]
        box[:2] = box[:2] - box[2:] / 2 + 1
        return box.astype(np.int32)

    @property
    def mahalanobis(self):
        return self.kf.mahalanobis

    @property
    def x(self):
        return self.state2box(self.kf.x)





