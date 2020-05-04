# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 15/4/2020

import numpy as np
import torch
import random


def set_random_seed(seed=26, deterministic=False, benchmark=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        benchmark (bool): Whether to keep input size same
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True
