# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 15/4/2020
import platform
if platform.system() == 'Windows':
    from blessed import Terminal
else:
    from blessings import Terminal
from progressbar import *
import sys
from tensorboardX import SummaryWriter
import csv


class TermLogger(object):
    """
        Terminal Logger for screen visualization
    """
    def __init__(self, n_epochs, train_size, valid_size=0, train_bar_size=2, valid_bar_size=2):
        """
        Parameters
        ----------
        n_epochs: int
            number of epochs
        train_size: int
            number of training iterations
        valid_size:int
            number of validation iterations
        train_bar_size: int
            height of training bar
        valid_bar_size: int
            height of valid bar

        Examples
        --------------
        Command Line: TermLogger(epochs=10,train_size=200,valid_size=100, train_bar_size=3, valid_bar_size=2)
        [Space]
        Epoch_bar Line
        [Space]
        Train_Writer Line

        Train_bar Line
        [Space]
        Valid_Writer Line
        Valid_bar Line
        [Space]
        """
        if valid_size > 0:
            assert valid_bar_size >= 2
        assert train_bar_size >= 2

        self.n_epochs = n_epochs
        self.train_size = train_size
        self.valid_size = valid_size
        self.terminal = Terminal()
        self.t_height = self.terminal.height

        epoch_bar_size = 1
        block_size = epoch_bar_size + train_bar_size + 3
        if valid_size > 0:
            block_size += valid_bar_size + 1
        block_begin = self.terminal.height - block_size - 1

        epoch_bar_begin = block_begin + epoch_bar_size
        train_writer_begin = epoch_bar_begin + 2
        train_bar_begin = train_writer_begin + train_bar_size - 1
        valid_writer_begin = train_bar_begin + 2
        valid_bar_begin = valid_writer_begin + valid_bar_size - 1

        # create block
        for i in range(block_size):
            print('')

        widgets = ['Epochs:', SimpleProgress(), ' ', Bar('#'),
                   ' ', Timer(), ' ', ETA(format='ETA: %(eta)8s')]
        self.epoch_bar = ProgressBar(widgets=widgets, max_value=n_epochs,
                                                 fd=Writer(self.terminal, (0, epoch_bar_begin)))  # epoch
        self.train_writer = Writer(self.terminal, (0, train_writer_begin))  # loss
        self.train_bar_writer = Writer(self.terminal, (0, train_bar_begin))  # batch
        self.reset_train_bar()
        if self.valid_size > 0:
            self.valid_writer = Writer(self.terminal, (0, valid_writer_begin))  # error
            self.valid_bar_writer = Writer(self.terminal, (0, valid_bar_begin))  # batch
            self.reset_valid_bar()

        self.epoch_bar.start()

    def reset_train_bar(self):
        widgets = ['Iters:', SimpleProgress(), ' ', Bar('#'),
                   ' ', Timer(), ' ', ETA(format='ETA: %(eta)8s')]
        self.train_bar = ProgressBar(widgets=widgets, max_value=self.train_size,
                                     fd=self.train_bar_writer)

    def reset_valid_bar(self):
        widgets = ['Iters:', SimpleProgress(), ' ', Bar('#'),
                   ' ', Timer(), ' ', ETA(format='ETA: %(eta)8s')]
        self.valid_bar = ProgressBar(widgets=widgets, max_value=self.valid_size,
                                     fd=self.valid_bar_writer)

    def reset_bar(self, train=True, valid=True):

        if train:
            self.reset_train_bar()

        if valid:
            self.reset_valid_bar()


class Writer(object):
    def __init__(self, t, location):
        self.terminal = t
        self.h = self.terminal.height
        self.x = location[0]
        self.y = location[1]

    def write(self, string):
        self.y += self.terminal.height - self.h
        self.h = self.terminal.height
        with self.terminal.location(self.x, self.y):
            sys.stdout.write('\033[K')
            print(string)

    def flush(self):
        return


class AverageMeter(object):
    """
        Record variables' value
        Output the average and summarize values
    """

    def __init__(self, n=1, precision=3):
        """
        Parameters
        ----------
        n: int
            number of metrics
        precision: int
            output precision for float
        """

        self.meters = n
        self.precision = precision
        self.reset(self.meters)

    def reset(self, n=1):
        self.val = [0]*n
        self.avg = [0]*n
        self.sum = [0]*n
        self.count = 0

    def update(self, val, n=1):
        """

        Parameters
        ----------
        val: float or int or list
        n: int
            number of values
        -------

        """
        if not isinstance(val, list):
            val = [val]

        assert (len(val) == self.meters)
        self.count += n
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)


