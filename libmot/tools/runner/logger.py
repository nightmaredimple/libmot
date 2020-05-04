# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 15/4/2020
import platform
if platform.system() == 'Windows':
    from blessed import Terminal
else:
    from blessings import Terminal
import torch
import sys
import time
import logging
import os.path as osp
from progressbar import *
from tensorboardX import SummaryWriter
from libmot.utils import mkdir_or_exist
from libmot.tools import Config
import shutil


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


def get_logger(log_file=None, log_level=logging.INFO):
    """Get the logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(__name__.split('.')[0])
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=log_file, filemode='w', format=format_str, level=log_level)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            '"silent" or None, but got {}'.format(type(logger)))


class LogManager(object):
    def __init__(self, log_path='log'):
        """
        Parameters
        ----------
        log_path: str
            it must be a dirname
        strict: bool
            if True, then the log path is log dir
            else: log dir = log path/current time
        """

        self.n_iter = 0
        self.train_iters = 0
        self.valid_iters = 0
        self.screen_displayer = None
        self.train_displayer = None
        self.valid_displayer = None

        t = time.localtime()
        dirname = time.strftime("%Y%m%d_%H_%M_%S", t)

        self.log_dir = osp.join(log_path, dirname)
        mkdir_or_exist(self.log_dir)
        self.log_file = osp.join(self.log_dir, 'log.txt')
        self.file_logger = get_logger(log_file=self.log_file, log_level=logging.INFO)

        self.checkpoint_dir = osp.join(self.log_dir, 'checkpoints')
        mkdir_or_exist(self.checkpoint_dir)

    def screen_logger(self, n_epochs, train_iters, valid_iters=0, train_bar_size=2, valid_bar_size=2):
        """
        Parameters
        ----------
        n_epochs: int
            number of epochs
        train_iters: int
            number of training iterations
        valid_iters:int
            number of validation iterations
        train_bar_size: int
            height of training bar
        valid_bar_size: int
            height of valid bar

        """
        self.n_epochs = n_epochs
        self.train_iters = train_iters
        self.valid_iters = valid_iters
        self.screen_displayer = TermLogger(n_epochs, train_iters, valid_iters, train_bar_size, valid_bar_size)

    def web_logger(self, is_train=True, is_valid=True):
        """
        logging for tensorboardX

        Parameters
        ----------
        is_train: bool
            whether to record during training
        is_valid: bool
            whether to record during training
        Returns
        -------

        """
        if is_train:
            self.train_displayer = SummaryWriter(osp.join(self.log_dir, 'train'))

        if is_valid:
            self.valid_displayer = SummaryWriter(osp.join(self.log_dir, 'val'))

    def stop(self):
        """
            Stop the screen displayer
        """
        self.screen_displayer.epoch_bar.finish()
        if self.train_displayer is not None:
            self.train_displayer.close()
        if self.valid_displayer is not None:
            self.valid_displayer.close()

    def write(self, msg):
        """
            Write messages to file logger
        """
        self.file_logger.info(msg)

    def save_checkpoint(self, epoch, state_dict, optimizer, lr_schedule, model_name, is_best=False):
        """
        Parameters
        ----------
        epoch: current epoch
        state_dict: model weights
        optimizer: optimizer state_dict()
        lr_schedule: lr_scheduler.state_dict()
        model_name: model name, used for naming checkpoint
        is_best: whether the performance of validation is the best in history
        """
        state = {'epoch': epoch,
                 'state_dict': state_dict,
                 'optimizer': optimizer,
                 'lr_schedule': lr_schedule}
        save_path = osp.join(self.checkpoint_dir, '{}_{}.pth.tar'.format(model_name, epoch))
        torch.save(state, save_path)
        if is_best:
            shutil.copyfile(save_path, osp.join(self.checkpoint_dir, '{}_best.pth.tar'.format(model_name)))

    def save_config(self, cfg):
        """Save current config,
            cfg must be Config instance or str or file path
        """
        if isinstance(cfg, Config):
            cfg = cfg.text

        if isinstance(cfg, str):
            if '.yaml' == cfg[-5:] or '.py' == cfg[-3:]:
                shutil.copyfile(cfg, osp.join(self.log_dir, osp.basename(cfg)))
            else:
                with open(osp.join(self.log_dir, 'config.txt'), 'w') as f:
                    f.write(cfg)



