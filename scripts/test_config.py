# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 17/4/2020
import os.path as osp
from libmot.tools import Config
import yaml

yaml_file = osp.join(osp.dirname(__file__), 'data\\faster_rcnn_R_101_FPN_3x.yaml')
yaml_cfg = Config.fromfile(yaml_file)
print(yaml_cfg)
print('---------------------------------------------')
python_file = osp.join(osp.dirname(__file__), 'data\\faster_rcnn_r101_fpn_1x.py')
python_cfg = Config.fromfile(python_file)
print(python_cfg)



