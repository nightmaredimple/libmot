# -*- coding: utf-8 -*-
# Author : Huang Piao
# Email  : huangpiao2985@163.com
# Date   : 6/11/2019

from __future__ import division
import numpy as np
import math

def reshape_z(z, dim_z, ndim):
    """ ensure z is a (dim_z, 1) shaped vector

    Parameters
    ----------
    z: array_like
        measurement for this update
    dim_z: int
        dims of observation variables
    ndim: int
        dims of state variables

    Returns
    ----------
    z: (dim_z, 1)
        (dim_z, 1) shaped vector

    """

    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    if z.shape != (dim_z, 1):
        raise ValueError('z must be convertible to shape ({}, 1)'.format(dim_z))

    if ndim == 1:
        z = z[:, 0]

    if ndim == 0:
        z = z[0, 0]

    return z

class KalmanFilter(object):

    def __init__(self, dim_x, dim_z, dim_u = 0, x = None, P = None,
                 Q = None, B = None, F = None, H = None, R = None):
        """Kalman Filter
        Refer to http:/github.com/rlabbe/filterpy

        Method
        -----------------------------------------
         Predict        |        Update
        -----------------------------------------
                        |  K = PH^T(HPH^T + R)^-1
        x = Fx + Bu     |  y = z - Hx
        P = FPF^T + Q   |  x = x + Ky
                        |  P = (1 - KH)P
        -----------------------------------------
        note: In update unit, here is a more numerically stable way: P = (I-KH)P(I-KH)' + KRK'

        Parameters
        ----------
        dim_x: int
            dims of state variables, eg:(x,y,vx,vy)->4
        dim_z: int
            dims of observation variables, eg:(x,y)->2
        dim_u: int
            dims of control variables,eg: a->1
            p = p + vt + 0.5at^2
            v = v + at
            =>[p;v] = [1,t;0,1][p;v] + [0.5t^2;t]a
        """

        assert dim_x >= 1, 'dim_x must be 1 or greater'
        assert dim_z >= 1, 'dim_z must be 1 or greater'
        assert dim_u >= 0, 'dim_u must be 0 or greater'

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # initialization
        # predict
        self.x = np.zeros((dim_x, 1)) if x is None else x                   # state
        self.P = np.eye(dim_x)  if P is None else P                         # uncertainty covariance
        self.Q = np.eye(dim_x)  if Q is None else Q                         # process uncertainty for prediction
        self.B = None if B is None else B                                   # control transition matrix
        self.F = np.eye(dim_x)  if F is None else F                         # state transition matrix

        # update
        self.H = np.zeros((dim_z, dim_x)) if H is None else H               # Measurement function for state->observation
        self.R = np.eye(dim_z)  if R is None else R                         # observation uncertainty
        self._alpha_sq = 1.                              # fading memory control
        self.z = np.array([[None] * self.dim_z]).T       # observation
        self.K = np.zeros((dim_x, dim_z))                # kalman gain
        self.y = np.zeros((dim_z, 1))                    # estimation error
        self.S = np.zeros((dim_z, dim_z))                # system uncertainty, S = HPH^T + R
        self.SI = np.zeros((dim_z, dim_z))               # inverse system uncertainty, SI = S^-1

        self.inv = np.linalg.inv
        self._mahalanobis = None                         # Mahalanobis distance of measurement
        self.latest_state = 'init'                       # last process name

    def predict(self, u = None, B = None, F = None, Q = None):
        """
        Predict next state (prior) using the Kalman filter state propagation equations:
                             x = Fx + Bu
                             P = fading_memory*FPF^T + Q

        Parameters
        ----------

        u : ndarray
            Optional control vector. If not `None`, it is multiplied by B
            to create the control input into the system.

        B : ndarray of (dim_x, dim_z), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.

        F : ndarray of (dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.

        Q : ndarray of (dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = F @ self.x + B @ u
        else:
            self.x = F @ self.x

        # P = fading_memory*FPF' + Q
        self.P = self._alpha_sq * (F @ self.P @ F.T) + Q
        self.latest_state = 'predict'

    def update(self, z, R = None, H = None):
        """
        Update Process, add a new measurement (z) to the Kalman filter.
                    K = PH^T(HPH^T + R)^-1
                    y = z - Hx
                    x = x + Ky
                    P = (1 - KH)P or P = (I-KH)P(I-KH)' + KRK'

        If z is None, nothing is computed.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        R : ndarray, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : ndarray, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.y = np.zeros((self.dim_z, 1))
            return

        z = reshape_z(z, self.dim_z, self.x.ndim)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        if H is None:
            H = self.H

        if self.latest_state == 'predict':
            # common subexpression for speed
            PHT = self.P @ H.T

            # S = HPH' + R
            # project system uncertainty into measurement space
            self.S = H @ PHT + R

            self.SI = self.inv(self.S)


            # K = PH'inv(S)
            # map system uncertainty into kalman gain
            self.K = PHT @ self.SI

            # P = (I-KH)P(I-KH)' + KRK'
            # This is more numerically stable and works for non-optimal K vs
            # the equation P = (I-KH)P usually seen in the literature.
            I_KH = np.eye(self.dim_x) - self.K @ H
            self.P = I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T


        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - H @ self.x

        self._mahalanobis = math.sqrt(float(self.y.T @ self.SI @ self.y))
        #print(self.y, self.SI,self._mahalanobis)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain

        self.x = self.x + self.K @ self.y
        self.latest_state = 'update'

    def batch_filter(self, zs, Fs = None, Qs = None, Hs = None,
                     Rs = None, Bs = None, us = None, update_first = False):
        """ Batch processes a sequences of measurements.
            eg: there are many possible measurements for current state

        Parameters
        ----------

        zs : list-like
            list of measurements at each time step `self.dt`. Missing
            measurements must be represented by `None`.

        Fs : None, list-like, default=None
            optional value or list of values to use for the state transition
            matrix F.

            If Fs is None then self.F is used for all epochs.

            Otherwise it must contain a list-like list of F's, one for
            each epoch.  This allows you to have varying F per epoch.

        Qs : None, np.array or list-like, default=None
            optional value or list of values to use for the process error
            covariance Q.

            If Qs is None then self.Q is used for all epochs.

            Otherwise it must contain a list-like list of Q's, one for
            each epoch.  This allows you to have varying Q per epoch.

        Hs : None, np.array or list-like, default=None
            optional list of values to use for the measurement matrix H.

            If Hs is None then self.H is used for all epochs.

            If Hs contains a single matrix, then it is used as H for all
            epochs.

            Otherwise it must contain a list-like list of H's, one for
            each epoch.  This allows you to have varying H per epoch.

        Rs : None, np.array or list-like, default=None
            optional list of values to use for the measurement error
            covariance R.

            If Rs is None then self.R is used for all epochs.

            Otherwise it must contain a list-like list of R's, one for
            each epoch.  This allows you to have varying R per epoch.

        Bs : None, np.array or list-like, default=None
            optional list of values to use for the control transition matrix B.

            If Bs is None then self.B is used for all epochs.

            Otherwise it must contain a list-like list of B's, one for
            each epoch.  This allows you to have varying B per epoch.

        us : None, np.array or list-like, default=None
            optional list of values to use for the control input vector;

            If us is None then None is used for all epochs (equivalent to 0,
            or no control input).

            Otherwise it must contain a list-like list of u's, one for
            each epoch.

       update_first : bool, optional, default=False
            controls whether the order of operations is update followed by
            predict, or predict followed by update. Default is predict->update.

        Returns
        -------

        means : np.array((n,dim_x,1))
            array of the state for each time step after the update. Each entry
            is an np.array. In other words `means[k,:]` is the state at step
            `k`.

        covariance : np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the update.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        means_predictions : np.array((n,dim_x,1))
            array of the state for each time step after the predictions. Each
            entry is an np.array. In other words `means[k,:]` is the state at
            step `k`.

        covariance_predictions : np.array((n,dim_x,dim_x))
            array of the covariances for each time step after the prediction.
            In other words `covariance[k,:,:]` is the covariance at step `k`.

        mahalanobis: np.array((n,1))
            array of the mahalanobises for each time step during the update

        Examples
        --------

        this example demonstrates tracking a measurement where the time
        between measurement varies, as stored in dts. This requires
        that F be recomputed for each epoch. The output is then smoothed
        with an RTS smoother.

        zs = [t + np.random.randn()*4 for t in range (40)]
        Fs = [np.array([[1., dt], [0, 1]] for dt in dts]

        (mu, cov, _, _, _) = kf.batch_filter(zs, Fs=Fs)
        (xs, Ps, Ks) = kf.rts_smoother(mu, cov, Fs=Fs)
        """

        n = np.size(zs, 0)
        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n
        if Hs is None:
            Hs = [self.H] * n
        if Rs is None:
            Rs = [self.R] * n
        if Bs is None:
            Bs = [self.B] * n
        if us is None:
            us = [0] * n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = np.zeros((n, self.dim_x))
            means_p = np.zeros((n, self.dim_x))
        else:
            means = np.zeros((n, self.dim_x, 1))
            means_p = np.zeros((n, self.dim_x, 1))

        # state covariances from Kalman Filter
        covariances = np.zeros((n, self.dim_x, self.dim_x))
        covariances_p = np.zeros((n, self.dim_x, self.dim_x))
        mahalanobis = np.zeros((n, 1))

        if update_first:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.update(z, R = R, H = H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P
                mahalanobis[i, :] = self._mahalanobis

                self.predict(u = u, B = B, F = F, Q = Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P
        else:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.predict(u = u, B = B, F = F, Q = Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                self.update(z, R = R, H = H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P
                mahalanobis[i, :] = self._mahalanobis

        return (means, covariances, means_p, covariances_p, mahalanobis)

    def rts_smoother(self, Xs, Ps, Fs = None, Qs = None):
        """
        Runs the Rauch-Tung-Striebal Kalman smoother on a set of
        means and covariances computed by a Kalman filter. The usual input
        would come from the output of `KalmanFilter.batch_filter()`.

        Parameters
        ----------

        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        Fs : list-like collection of numpy.array, optional
            State transition matrix of the Kalman filter at each time step.
            Optional, if not provided the filter's self.F will be used

        Qs : list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used

        Returns
        -------

        x : numpy.ndarray
           smoothed means

        P : numpy.ndarray
           smoothed state covariances

        K : numpy.ndarray
            smoother gain at each step

        Pp : numpy.ndarray
           Predicted state covariances

        Examples
        --------

            zs = [t + np.random.randn()*4 for t in range (40)]

            (mu, cov, _, _, _) = kalman.batch_filter(zs)
            (x, P, K, Pp) = rts_smoother(mu, cov, kf.F, kf.Q)

        """

        if len(Xs) != len(Ps):
            raise ValueError('length of Xs and Ps must be the same')

        n = Xs.shape[0]
        dim_x = Xs.shape[1]

        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n

        # smoother gain
        K = np.zeros((n, dim_x, dim_x))

        x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()
        for k in range(n - 2, -1, -1):
            Pp[k] = Fs[k + 1] @ P[k] @ Fs[k + 1].T + Qs[k + 1]

            K[k] = P[k] @ Fs[k + 1].T @ self.inv(Pp[k])
            x[k] += K[k] @ (x[k + 1] - Fs[k + 1] @ x[k])
            P[k] += (K[k] @ (P[k + 1] - Pp[k])) @ K[k].T

        return (x, P, K, Pp)

    def get_prediction(self, u = 0):
        """
        Predicts the next state of the filter and returns it without
        altering the state of the filter.

        Parameters
        ----------

        u : np.array
            optional control input

        Returns
        -------

        (x, P) : tuple
            State vector and covariance array of the prediction.
        """
        if u == 0 :
            return (self.x, self.P)

        x = self.F @ self.x + self.B @ u
        P = self._alpha_sq * (self.F @ self.P @ self.F.T) + self.Q
        return (x, P)

    def get_update(self, z = None):
        """
        Computes the new estimate based on measurement `z` and returns it
        without altering the state of the filter.

        Parameters
        ----------

        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        Returns
        -------

        (x, P) : tuple
            State vector and covariance array of the update.
       """

        if z is None:
            return self.x, self.P
        z = reshape_z(z, self.dim_z, self.x.ndim)

        R = self.R
        H = self.H
        P = self.P
        x = self.x

        # error (residual) between measurement and prediction
        y = z - (H @ x)

        # common subexpression for speed
        PHT = P @ H.T

        # project system uncertainty into measurement space
        S = (H @ PHT) + R

        # map system uncertainty into kalman gain
        K = PHT @ self.inv(S)

        # predict new x with residual scaled by the kalman gain
        x = x + (K @ y)

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = np.eye(self.dim_x) - (K @ H)
        P = (I_KH @ P @ I_KH.T) + (K @ R @ K.T)

        return x, P

    def residual_of(self, z):
        """
        Returns the residual for the given measurement (z)
        Parameters
        ----------

        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        Returns
        -------

        y : numpy.array
            the residual for the given measurement (z)
       """
        z = reshape_z(z, self.dim_z, self.x.ndim)
        return z - self.H @ self.x

    def measurement_of_state(self, x):
        """
        Helper function that converts a state into a measurement.

        Parameters
        ----------

        x : np.array
            kalman state vector

        Returns
        -------

        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
        """

        return self.H @ x

    @property
    def mahalanobis(self):
        """"
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """

        if self._mahalanobis is None:
            self._mahalanobis = math.sqrt(float(self.y.T @self.SI @ self.y))
        return self._mahalanobis

    @property
    def alpha(self):
        """
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.
        """
        return self._alpha_sq ** .5

    @alpha.setter
    def alpha(self, value):
        if not np.isscalar(value) :
            raise ValueError('alpha must be a float ')

        self._alpha_sq = value ** 2

