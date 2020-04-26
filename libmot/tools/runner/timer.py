# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 14/4/2020
# referring to https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/timer.py
import time


class TimerError(Exception):

    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)


class Timer(object):
    def __init__(self, start=False, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}s'
        self._t_start = time.time()
        self._t_last = time.time()
        if start:
            self.tic()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.tic()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last()))
        self._is_running = False

    def tic(self):
        """Start the timer."""
        if not self._is_running:
            self._is_running = True
        self._t_start = time.time()
        self._t_last = time.time()

    def toc(self):
        """Total time since the timer is started.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time.time()
        return self._t_last - self._t_start

    def since_last(self):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.

        Returns (float): Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')

        dur = time.time() - self._t_last
        self._t_last = time.time()
        return dur


if __name__ == '__main__':
    T = Timer()
    T.tic()
    time.sleep(1)
    print('Since Begin: {:.3f}s'.format(T.toc()))
    time.sleep(2)
    print('Since Last: {:.3f}s'.format(T.since_last()))
    print('Since Begin: {:.3f}s'.format(T.toc()))

    with T as t:
        time.sleep(2)


