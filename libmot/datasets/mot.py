# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 19/4/2020
import os.path as osp
import numpy as np
import configparser
import cv2
from libmot.utils import check_file_exist, check_folder_exist

class MOTReader(object):
    def __init__(self, mot_folder=None, length=-1, image_folder=None, detection_file=None,
                 gt_file=None, info_file=None, detection_thresh=None, vis_thresh=None):
        """
        Parameters
        ----------
        mot_folder: str
            eg：./MOT-02-DPM/
        length: int
            tracked length, -1 means track all images
        image_folder：str
            eg: ./MOT-02-DPM/img1/
        detection_file：str
            eg: ./MOT-02-DPM/det/det.txt
        gt_file：str
            eg: ./MOT-02-DPM/gt/gt.txt
        info_file：str
            eg: ./MOT-02-DPM/seqinfo.ini
        detection_thresh: float
            threshold of detections, eg: 0.7
        vis_thresh: float
            threshold of visibility ratio in gt file, eg:0.3
        """
        assert mot_folder is not None or (image_folder is not None and detection_file is not None)

        if mot_folder is not None:
            mot_folder = osp.abspath(osp.expanduser(mot_folder))
            self.image_folder = osp.join(mot_folder, 'img1')
            self.detection_file = osp.join(mot_folder, 'det/det.txt')
            self.gt_file = osp.join(mot_folder, 'gt/gt.txt')
            self.info_file = osp.join(mot_folder, 'seqinfo.ini')
            self._check_dir([mot_folder, self.image_folder])
            self._check_file([self.detection_file, self.info_file])
        else:
            self.image_folder = osp.abspath(osp.expanduser(image_folder))
            self.detection_file = detection_file
            self.gt_file = gt_file
            self.info_file = info_file
            self._check_dir([self.image_folder])
            self._check_file([self.detection_file])

        self.video_info = self.get_video_information(self.info_file)
        self.vis_thresh = vis_thresh
        self.detection_thresh = detection_thresh
        self.image_format = osp.join(self.image_folder, '{0:06d}.jpg')
        self.track_length = self.video_info['length']
        if length > -1:
            self.track_length = length

        self.filter_gt(self.gt_file)
        self.filter_detection(self.detection_file)

    def _check_dir(self, pathlist):
        """Check all folders in pathlist, if not exist, raise error
        """
        assert isinstance(pathlist, list) or isinstance(pathlist, tuple)
        for path in pathlist:
            check_folder_exist(path)

    def _check_file(self, pathlist):
        """Check all files in pathlist, if not exist, raise error
        """
        assert isinstance(pathlist, list) or isinstance(pathlist, tuple)
        for path in pathlist:
            check_file_exist(path)

    def get_video_information(self, filepath):
        """get sequence information from seqinfo.ini"""
        video_info = {}
        if filepath is None or not osp.isfile(filepath):
            self.image_list = sorted(os.listdir(self.image_folder))
            img = cv2.imread(os.path.join(self.image_folder, self.image_list[0]))
            video_info.update({'shape': (img.shape[1], img.shape[0])})
            video_info.update({'length': len(self.image_list)})
            if osp.basename(self.image_folder) == 'img1':
                video_info.update({'name': osp.basename(osp.dirname(self.image_folder))})
            else:
                video_info.update({'name': osp.basename(self.image_folder)})
        else:
            seq_info = configparser.ConfigParser()
            seq_info.read(filepath)
            video_info = {"name": seq_info.get('Sequence', 'name'),
                          "shape": (int(seq_info.get('Sequence', 'imWidth')),
                                    int(seq_info.get('Sequence', 'imHeight'))),
                          "length": int(seq_info.get('Sequence', 'seqLength'))}

        return video_info

    def filter_gt(self, gt_file):
        """Filter rules are as follow:
            1.frame id <= track length
            2.is_active=True
            3.classID=1, note that classID 2 and 7 are not considered in the offical code
            4.visibility ratio > thresh

        Parameters
        ----------
        gt_file: str
            filepath of ground truth

        """
        if gt_file is not None and osp.isfile(gt_file):
            gt = np.genfromtxt(gt_file, delimiter=',')
            length_filter = gt[:, 0] <= self.track_length
            active_filter = gt[:, 6] == 1
            class_filter = (gt[:, 7] == 1)  | (gt[:, 7] == -1)
            gt = gt[length_filter & active_filter & class_filter]

            if self.vis_thresh is not None and (gt[:, -1] >= 0).any():
                gt = gt[gt[:, -1] >= self.vis_thresh]

            self.ground_truth = gt
        else:
            self.ground_truth = None

    def filter_detection(self, detection_file):
        """Filter rules are as follow:
            1.frame id <= track length
            2.confidence >= thresh

        Parameters
        ----------
        detection_file:str
            filepath of ground truth

        """
        dets = np.genfromtxt(detection_file, delimiter=',')
        dets = dets[dets[:, 0] <= self.video_info['length']]
        if self.detection_thresh is not None:
            dets = dets[dets[:, 6] >= self.detection_thresh]
        scores = np.c_[dets[:, 0], dets[:, 6]]

        dets = dets.astype(np.int32)
        self.detection = {'bboxes': {}, 'scores': {}}

        for i in range(self.track_length):
            self.detection['bboxes'].update({i: dets[dets[:, 0] == i+1, 2:6]})
            self.detection['scores'].update({i: scores[scores[:, 0] == i+1, 1]})

    def get_image(self, index, strict=False):
        if strict:
            assert index < self.track_length, "index should be lower than track length"
        else:
            assert index < self.video_info['length'], "index should be lower than sequence length"

        return cv2.imread(self.image_format.format(index))

    def __len__(self):
        return self.track_length








