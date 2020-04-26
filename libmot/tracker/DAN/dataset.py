# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 21/4/2020
# referring to https://github.com/shijieS/SST/blob/master/data/mot.py

import os
import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd
import random


class Node:
    def __init__(self, box, frame_id, next_frame_id=-1):
        """Traget Node: box, frame id and next frame id
            next frame id = -1 means this is the last node in the target trajectory
        """
        self.box = box
        self.frame_id = frame_id
        self.next_frame_id = next_frame_id


class Track:
    def __init__(self, id):
        """trajectory of each target id
        """
        self.nodes = list()
        self.id = id

    def add_node(self, n):
        if len(self.nodes) > 0:
            self.nodes[-1].next_frame_id = n.frame_id
        self.nodes.append(n)

    def get_node_by_index(self, index):
        return self.nodes[index]


class Tracks:
    def __init__(self):
        """All targets' trajectories list
        """
        self.tracks = list()

    def add_node(self, node, id):
        node_add = False
        track_index = 0
        node_index = 0
        for t in self.tracks:
            if t.id == id:
                t.add_node(node)
                node_add = True
                track_index = self.tracks.index(t)
                node_index = t.nodes.index(node)
                break
        if not node_add:
            t = Track(id)
            t.add_node(node)
            self.tracks.append(t)
            track_index = self.tracks.index(t)
            node_index = t.nodes.index(node)

        return track_index, node_index

    def get_track_by_index(self, index):
        return self.tracks[index]


class GTSingleParser:
    def __init__(self, folder, min_visibility=0.3, min_gap=0, max_gap=30):
        """ Sampler of one sequence

        Parameters
        ----------
        folder: sequence folder,eg:./MOT-02-DPM/
        min_visibility: visibility threshold of gt bboxes
        min_gap: t-> t + random(next_frame_index)+min_gap
        max_gap: t-> t + random(next_frame_index)+max_gap
        """

        self.min_gap = min_gap
        self.max_gap = max_gap
        # 1. get the gt path and image folder
        gt_file_path = os.path.join(folder, 'gt/gt.txt')
        self.folder = folder

        # 2. read the gt data
        gt_file = pd.read_csv(gt_file_path, header=None)
        gt_file = gt_file[gt_file[6] == 1]
        gt_file = gt_file[gt_file[8] > min_visibility]
        gt_group = gt_file.groupby(0)
        gt_group_keys = gt_group.indices.keys()
        self.max_frame_index = max(gt_group_keys)
        # 3. update tracks
        self.tracks = Tracks()
        self.recorder = {}
        for key in gt_group_keys:
            det = gt_group.get_group(key).values
            ids = np.array(det[:, 1]).astype(int)
            det = np.array(det[:, 2:6])
            det[:, 2:4] += det[:, :2]

            self.recorder[key - 1] = list()
            # 3.1 update tracks
            for id, d in zip(ids, det):
                node = Node(d, key - 1)
                track_index, node_index = self.tracks.add_node(node, id)
                self.recorder[key - 1].append((track_index, node_index))

    def _getimage(self, frame_index):
        image_path = os.path.join(self.folder, 'img1/{0:06}.jpg'.format(frame_index))
        return cv2.imread(image_path)

    def get_item(self, frame_index):
        '''
        get the current_image, current_boxes, next_image, next_boxes, labels from the frame_index
        :param frame_index:
        :return: current_image, current_boxes, next_image, next_boxes, labels
        '''
        if frame_index not in self.recorder:
            return None, None, None, None, None
        # get current_image, current_box, next_image, next_box and labels
        current_image = self._getimage(frame_index)
        current_boxes = list()
        current = self.recorder[frame_index]
        next_frame_indexes = list()
        current_track_indexes = list()
        # 1. get current box
        for track_index, node_index in current:
            t = self.tracks.get_track_by_index(track_index)
            n = t.get_node_by_index(node_index)
            current_boxes.append(n.box)

            current_track_indexes.append(track_index)
            if n.next_frame_id != -1:
                next_frame_indexes.append(n.next_frame_id)

        # 2. decide the next frame (0.5 probability to choose the farest ones,
        # and other probability to choose the frame between them)
        if len(next_frame_indexes) == 0:
            return None, None, None, None, None
        if len(next_frame_indexes) == 1:
            next_frame_index = next_frame_indexes[0]
        else:
            max_next_frame_index = max(next_frame_indexes)
            is_choose_farest = bool(random.getrandbits(1))
            if is_choose_farest:
                next_frame_index = max_next_frame_index
            else:
                next_frame_index = random.choice(next_frame_indexes)
                gap_frame = random.randint(self.min_gap, self.max_gap)
                temp_frame_index = next_frame_index + gap_frame
                choice_gap = list(range(self.min_gap, self.max_gap))
                if self.min_gap != 0:
                    choice_gap.append(0)
                while temp_frame_index not in self.recorder:
                    gap_frame = random.choice(choice_gap)
                    temp_frame_index = next_frame_index + gap_frame
                next_frame_index = temp_frame_index

        # 3. get next image
        next_image = self._getimage(next_frame_index)

        # 4. get next frame boxes
        next = self.recorder[next_frame_index]
        next_boxes = list()
        next_track_indexes = list()
        for track_index, node_index in next:
            t = self.tracks.get_track_by_index(track_index)
            next_track_indexes.append(track_index)
            n = t.get_node_by_index(node_index)
            next_boxes.append(n.box)

        # 5. get the labels
        current_track_indexes = np.array(current_track_indexes)
        next_track_indexes = np.array(next_track_indexes)
        labels = np.repeat(np.expand_dims(np.array(current_track_indexes), axis=1), len(next_track_indexes),
                           axis=1) == np.repeat(np.expand_dims(np.array(next_track_indexes), axis=0),
                                                len(current_track_indexes), axis=0)

        # 6. return all values
        # 6.1 change boxes format
        current_boxes = np.array(current_boxes)
        next_boxes = np.array(next_boxes)
        # 6.2 return the corresponding values
        return current_image, current_boxes, next_image, next_boxes, labels

    def __len__(self):
        return self.max_frame_index


class GTParser:
    def __init__(self, mot_root, detector='FRCNN', valid=None, phase='train'):
        """
        Parameters
        ----------
        mot_root: mot foler, eg: ./MOT17/
        detector: DPM/FRCNN/SDP or ''(this is for MOT15)
        valid: filter out some sequences, eg: ['10','11']
        phase:train or valid
        """
        # analsis all the folder in mot_root
        # 1. get all the folders
        mot_root = os.path.join(mot_root, 'train')
        if valid is None:
            all_folders = sorted(
                [os.path.join(mot_root, i) for i in os.listdir(mot_root)
                 if os.path.isdir(os.path.join(mot_root, i))
                 and i.find(detector) != -1]
            )
        else:
            all_folders = []
            for seq in os.listdir(mot_root):
                if os.path.isdir(os.path.join(mot_root, seq)) and detector in seq:
                    flag = True if phase == 'train' else False
                    for ex in valid:
                        if phase == 'train' and ex in seq:
                            flag &= False
                        if phase == 'valid' and ex in seq:
                            flag |= True

                    if flag:
                        all_folders.append(os.path.join(mot_root, seq))

        # 2. create single parser
        self.parsers = [GTSingleParser(folder) for folder in sorted(all_folders)]

        # 3. get some basic information
        self.lens = [len(p) for p in self.parsers]
        self.len = sum(self.lens)

    def __len__(self):
        # get the length of all the matching frame
        return self.len

    def __getitem__(self, item):
        """ item size = sum of len(all folders' samples)
        """
        if item < 0:
            return None, None, None, None, None
        # 1. find the parser
        total_len = 0 # accumulative length
        index = 0  # sequence index
        current_item = item  # frame index in sequence
        for l in self.lens:
            total_len += l
            if item < total_len:
                break
            else:
                index += 1
                current_item -= l

        # 2. get items
        if index >= len(self.parsers):
            return None, None, None, None, None
        return self.parsers[index].get_item(current_item)


class MOTTrainDataset(data.Dataset):
    '''
    The class is the dataset for train, which read gt.txt file and rearrange them as the tracks set.
    it can be selected from the specified frame
    '''
    def __init__(self, cfg, transform=None, phase='train'):
        """
        Parameters
        ----------
        cfg: contains at least:
            mot_root: eg: ./MOT17
            detector: DPM/FRCNN/SDP,
            max_object: max number of objects between two frames
            max_gap: t-> t + random(next_frame_index)+max_gap
            valid: list, filter out some sequences, eg: ['10','11']
        transform： transform for augmentation
        phase: train/valid
        """
        # 1. init all the variables
        self.mot_root = cfg['io']['mot_root']
        self.transform = transform
        self.phase = phase
        self.detector = cfg['io']['detector']
        self.max_object = cfg['datasets']['max_object']
        self.max_gap = cfg['datasets']['max_gap']
        self.valid = cfg['io']['valid']

        # 2. init GTParser
        self.parser = GTParser(self.mot_root, self.detector, self.valid, self.phase)

    def __getitem__(self, item):
        current_image, current_box, next_image, next_box, labels = self.parser[item]

        # item index is negative/out of bound, or all targets of the chosen frame are the end of the trajectories
        while current_image is None:
            current_image, current_box, next_image, next_box, labels = \
                self.parser[item+random.randint(-self.max_gap, self.max_gap)]

        if self.transform is None:
            return current_image, current_box, next_image, next_box, labels

        # change the label to max_object x max_object
        labels = np.pad(labels,
                        [(0, self.max_object - labels.shape[0]),
                         (0, self.max_object - labels.shape[1])],
                        mode='constant',
                        constant_values=0)
        return self.transform(current_image, next_image, current_box, next_box, labels)

    def __len__(self):
        return len(self.parser)


