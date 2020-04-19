# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 15/4/2020
# referring to https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/path.py

import os
import os.path as osp
from pathlib import Path


def is_str(x):
    """bool: indicate whether the x is string"""
    return isinstance(x, str)


def is_path(x):
    """bool: indicate whether the x is string or Path"""
    return is_str(x) or isinstance(x, Path)


def fopen(filepath, *args, **kwargs):
    """Open file"""
    assert is_path(filepath), "filepath is illegal"

    if is_str(filepath):
        return open(filepath, *args, **kwargs)
    elif isinstance(filepath, Path):
        return filepath.open(*args, **kwargs)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    """Check the file path"""
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

def check_folder_exist(foldername, msg_tmpl='folder "{}" does not exist'):
    """Check the folder path"""
    if not osp.isdir(foldername):
        raise FileNotFoundError(msg_tmpl.format(foldername))


def mkdir_or_exist(dir_name, mode=0o777):
    """Create the directory recursively"""
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src, dst, overwrite=True, **kwargs):
    """Make soft link from src to dst"""
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def scandir(dir_path, suffix=None, blacklists=None, recursive=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        blacklists (str | tuple(str), optional): blacklists of some string
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if (blacklists is not None) and not isinstance(blacklists, (str, tuple)):
        raise TypeError('"blacklists" must be a string or tuple of strings')

    root = dir_path

    def in_blacklists(filepath, blacklists):
        if blacklists is None:
            return False

        if is_str(blacklists):
            blacklists = (blacklists,)

        for s in blacklists:
            if s in filepath:
                return True
        return False

    def _scandir(dir_path, suffix, recursive):
        if not osp.isdir(dir_path):
            return
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if in_blacklists(rel_path, blacklists):
                    continue

                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


if __name__ == '__main__':
    for p in scandir('..\\..', suffix='.py', blacklists='venv', recursive=True):
        print(p)
    #print(list(scandir('..\\..', suffix='.py', recursive=True)))
