# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 21/4/2020
# referring to https://github.com/shijieS/SST/blob/master/layer/sst.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DAN(nn.Module):
    def __init__(self, base, extras, selector, final_net, cfg):
        """DAN Network

        Parameters
        ----------
        base: vgg16
        extras:extension
        selector: selected features
        final_net: final net
        cfg: config dict
        """
        super(DAN, self).__init__()

        # vgg network
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.selector = nn.ModuleList(selector)

        self.stacker2_bn = nn.BatchNorm2d(int(cfg['model']['final_net'][str(cfg['datasets']['image_size'])][0] / 2))
        self.final_dp = nn.Dropout(0.5)
        self.final_net = nn.ModuleList(final_net)

        self.image_size = cfg['datasets']['image_size']
        self.max_object = cfg['datasets']['max_object']
        self.selector_channel = cfg['model']['selector_channel']

        self.false_objects_column = None
        self.false_objects_row = None
        self.false_constant = cfg['datasets']['false_constant']
        self.cfg = cfg

    def forward(self, img_pre=None, img_next=None, box_pre=None, box_next=None, cache=None):
        """

        Parameters
        ----------
        img_pre: the previous image, (1, 3, 900, 900)
        img_next: the next image,  (1, 3, 900, 900)
        box_pre：the previous box center, (1, 80, 1, 1, 2)
        box_next：the next box center, (1, 80, 1, 1, 2)
        cache: the previous features

        Returns
        -------
        the similarity matrix
        """

        if cache is not None:
            feature_pre = cache
        else:
            feature_pre = self.forward_feature_extracter(img_pre, box_pre)
        feature_next = self.forward_feature_extracter(img_next, box_next) # [B,N,C]

        # [B, C, N, N]
        x = self.forward_stacker2(feature_pre, feature_next)
        x = self.final_dp(x)

        # [B, 1, N, N]
        x = self.forward_final(x, self.final_net)

        # add false unmatched row and column
        x = self.add_unmatched_dim(x)

        return x

    def forward_feature_extracter(self, img, box):
        """
        Parameters
        ----------
        img : input image [B,C,H,W]
        box : detection centers, [B,N,1,1,2]

        Returns
        ----------
        feature: [B,N,C]
        """

        sources = list()
        vgg = self.forward_vgg(img, self.vgg, sources)
        extra = self.forward_extras(vgg, self.extras, sources)
        feature = self.forward_selector_stacker1(sources, box, self.selector)

        return feature

    def get_similarity(self, image1=None, detection1=None, image2=None, detection2=None, cache=None):
        """Get Affinity of bboxes between two frames

        Parameters
        ----------
        image1: pre image
        detection1: pre centers of boxes
        image2: next image
        detection2: next centers of boxes
        cache: use pre features to save time

        Returns
        -------

        """
        if cache is not None:
            feature1 = cache
        else:
            feature1 = self.forward_feature_extracter(image1, detection1)
        feature2 = self.forward_feature_extracter(image2, detection2)
        return self.forward_stacker_features(feature1, feature2, False)

    def forward_stacker_features(self, xp, xn, fill_up_column=True):
        """Get Affinity of features

        Parameters
        ----------
        xp: pre features, [B,Nm,C]
        xn: next_feature, [B,Nn,C]
        fill_up_column:bool

        Returns
        -------
        affinity: [Nmx(Nn+1)]
        """

        pre_rest_num = self.max_object - xp.shape[1]
        next_rest_num = self.max_object - xn.shape[1]
        pre_num = xp.shape[1]
        next_num = xn.shape[1]
        # xp/xn: [B,N,C]
        x = self.forward_stacker2(
            F.pad(xp, pad=[0, 0, 0, pre_rest_num], mode='constant', value=0),
            F.pad(xn, pad=[0, 0, 0, next_rest_num], mode='constant', value=0)
        )

        x = self.final_dp(x)
        # [B=1, C=1, N, N]
        x = self.forward_final(x, self.final_net)
        x = x.contiguous()
        # add zero
        if next_num < self.max_object:
            x[0, 0, :, next_num:] = 0
        if pre_num < self.max_object:
            x[0, 0, pre_num:, :] = 0
        x = x[0, 0, :]  # [N,N]
        # add false unmatched row and column
        x = F.pad(x, pad=[0, 1, 0, 1], mode='constant', value=self.false_constant)  # [N+1, N+1]

        x_f = F.softmax(x, dim=1)
        x_t = F.softmax(x, dim=0)

        # slice
        last_row, last_col = x_f.shape
        row_slice = list(range(pre_num)) + [last_row - 1]   # [0,...Nm-1,N-1]
        col_slice = list(range(next_num)) + [last_col - 1]  # [0,...Nn-1,N-1]
        x_f = x_f[row_slice, :]
        x_f = x_f[:, col_slice]   # [Nm+1, Nn+1]
        x_t = x_t[row_slice, :]
        x_t = x_t[:, col_slice]   # [Nm+1, Nn+1]

        x = torch.zeros(pre_num, next_num + 1)  # [Nmx(Nn+1)]
        # x[0:pre_num, 0:next_num] = torch.max(x_f[0:pre_num, 0:next_num], x_t[0:pre_num, 0:next_num])
        # x[:Nm, :Nn] = (xf[:Nm, :Nn] + xt[:Nm, :Nn])/2
        # x[:, Nn] = xf[:, Nn]
        x[0:pre_num, 0:next_num] = (x_f[0:pre_num, 0:next_num] + x_t[0:pre_num, 0:next_num]) / 2.0
        x[:, next_num:next_num+1] = x_f[:pre_num, next_num:next_num+1]
        if fill_up_column and pre_num > 1:
            x = torch.cat([x, x[:, -1].repeat(1, pre_num - 1)], dim=1)  # [Nmx(Nn+Nm)]

        return x.detach().cpu().numpy()

    def forward_vgg(self, x, vgg, sources):
        for k in range(16):
            x = vgg[k](x)
        sources.append(x)

        for k in range(16, 23):
            x = vgg[k](x)
        sources.append(x)

        for k in range(23, 35):
            x = vgg[k](x)
        sources.append(x)
        return x

    def forward_extras(self, x, extras, sources):
        for k, v in enumerate(extras):
            x = v(x)
            if k % 6 == 3:  # conv + batch norm = 2
                sources.append(x)
        return x

    def forward_selector_stacker1(self, sources, labels, selector):
        """

        Parameters
        ----------
        sources: sources of selectors,[B, C, H, W]
        labels: centers of detections, [B, N, 1, 1, 2], range[-1,1]
        selector: selectors dropped behind the selected layers

        Returns
        -------
        features: [B, N, C]
        """

        sources = [
            F.relu(net(x), inplace=True) for net, x in zip(selector, sources)
        ]

        res = list()
        for label_index in range(labels.size(1)):
            label_res = list()
            for source_index in range(len(sources)):
                label_res.append(
                    F.grid_sample(sources[source_index],     # [B, Ci, H, W]
                                  labels[:, label_index, :],  # [B, 1, 1, 2]
                                  align_corners=True).squeeze(2).squeeze(2)    # [B, Ci]
                )                                            # Fni x [B,Ci]
            res.append(torch.cat(label_res, 1))              # Nm x [B, sum(CixFni)],  eg :C=sum(CixFni)= 520

        return torch.stack(res, 1)                           # [B, Nm, C]

    def forward_stacker2(self, stacker1_pre_output, stacker1_next_output):
        # Nm=Nn=max_object
        # [B, Nn, C]->[B, Nn, 1, C]->[B, Nn, max_object, C]->[B,C,Nn,max_object]
        stacker1_pre_output = stacker1_pre_output.unsqueeze(2).repeat(1, 1, self.max_object, 1).permute(0, 3, 1, 2)
        # [B, Nm, C]->[B, 1, Nm, C]->[B, max_object, Nm, C]->[B,C,max_object,Nm]
        stacker1_next_output = stacker1_next_output.unsqueeze(1).repeat(1, self.max_object, 1, 1).permute(0, 3, 1, 2)

        stacker1_pre_output = self.stacker2_bn(stacker1_pre_output.contiguous())
        stacker1_next_output = self.stacker2_bn(stacker1_next_output.contiguous())

        output = torch.cat([stacker1_pre_output, stacker1_next_output], 1)  # [B, 2C, max_object, max_object]

        return output

    def forward_final(self, x, final_net):
        x = x.contiguous()
        for f in final_net:
            x = f(x)
        return x

    def add_unmatched_dim(self, x):
        """[N,M]->[N+1,M+1] with padding false_constant
        """
        if self.false_objects_column is None:
            self.false_objects_column = torch.ones(x.shape[0], x.shape[1], x.shape[2], 1, device=x.device) * self.false_constant
        x = torch.cat([x, self.false_objects_column], 3)

        if self.false_objects_row is None:
            self.false_objects_row = torch.ones(x.shape[0], x.shape[1], 1, x.shape[3], device=x.device) * self.false_constant
        x = torch.cat([x, self.false_objects_row], 2)
        return x


def build_vgg(cfg, in_channels=3, batch_norm=False):
    """Build VGG-19
    Parameters
    ----------
    cfg: structure of vgg
     eg:[64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                'C', 512, 512, 512, 'M', 512, 512, 512]
    in_channels： input channels
    batch_norm: whether using batch normalization

    """
    layers = []
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, in_channels, batch_norm=True):
    """Build Extension
    Parameters
    ----------
    cfg: structure of extensions
     eg:[256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256,
                128, 'S', 256, 128, 256]
    in_channels： input channels
    batch_norm: whether using batch normalization

    """
    layers = []
    in_channels = in_channels
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                conv2d = nn.Conv2d(in_channels, cfg[k+1], kernel_size=(1, 3)[flag],
                                   stride=2, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[k+1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            flag = not flag
        in_channels = v
    return layers


def add_final(cfg, batch_norm=True):
    """Build Final
    Parameters
    ----------
    cfg: structure of final net
     eg:[1040, 512, 256, 128, 64, 1]
    batch_norm: whether using batch normalization

    """
    layers = []
    in_channels = int(cfg[0])
    layers += []
    # 1. add the 1:-2 layer with BatchNorm
    for v in cfg[1:-2]:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    # 2. add the -2: layer without BatchNorm for BatchNorm would make the output value normal distribution.
    for v in cfg[-2:]:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=1, stride=1)
        layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v

    return layers


def build_selector(vgg, extra_layers, vgg_selector, selector_channels, batch_norm=True):
    """Select output layers for multi-level features(selected layer + conv=output)

    Parameters
    ----------
    vgg: vggnet
    extra_layers: extentions
    vgg_selector: selected layers' index, eg: [15, 25, -1]
    selector_channels: output channels of each selected layer
    batch_norm: batch_normal must be same to add_extras batch_normal
    """
    selector_layers = []

    for k, v in enumerate(vgg_selector):
         selector_layers += [nn.Conv2d(vgg[v-1].out_channels, selector_channels[k],
                                       kernel_size=3, padding=1)]
    if batch_norm:
        for k, v in enumerate(extra_layers[3::6], 3):
            selector_layers += [nn.Conv2d(v.out_channels, selector_channels[k],
                                          kernel_size=3, padding=1)]
    else:
        for k, v in enumerate(extra_layers[3::4], 3):
            selector_layers += [nn.Conv2d(v.out_channels, selector_channels[k],
                                          kernel_size=3, padding=1)]

    return vgg, extra_layers, selector_layers


def build_dan(cfg):
    """Build DAN-Net

    Parameters
    ----------
    cfg: contains
        size: input image size, eg: 900
        base_net: vgg structure, eg:{
            '900': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                    'C', 512, 512, 512, 'M', 512, 512, 512],
            '1024': [],}
        extra_net: extension structure,eg:{
            '900': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256,
                    128, 'S', 256, 128, 256],  # new: this line
            '1024': [],
        }
        final_net: final structure,eg:{
            '900': [1040, 512, 256, 128, 64, 1],
            '1024': []
        }

        vgg_selector: selected layers' index, eg: [15, 25, -1]
        selector_channels: output channels of each selected layer

    """
    size = cfg['datasets']['image_size']
    if size != 900:
        print('Error: Sorry only SST{} is supported currently!'.format(size))
        return

    base_net = cfg['model']['base_net']
    extra_net = cfg['model']['extra_net']
    final_net = cfg['model']['final_net']
    vgg_selector = cfg['model']['vgg_selector']
    selector_channels = cfg['model']['selector_channel']

    return DAN(*build_selector(
                   build_vgg(base_net[str(size)], 3),
                   add_extras(extra_net[str(size)], 1024),
                   vgg_selector,
                   selector_channels
               ),
               add_final(final_net[str(size)]),
               cfg
               )
