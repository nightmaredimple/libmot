# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 27/4/2020
# referring to https://github.com/shijieS/SST/blob/master/tracker.py

from libmot.tools import set_random_seed
from libmot.tracker.DAN import build_dan
from libmot.data_association import LinearAssignment
import torch
import cv2
import numpy as np


def get_iou(pre_boxes, next_boxes):
    """if two boxes are overlapped, then iou = A∩B/A∪B
       else, iou = -(min_rect(A,B)-A-B)/(A+B)
    """
    h = len(pre_boxes)
    w = len(next_boxes)
    if h == 0 or w == 0:
        return []

    iou = np.zeros((h, w), dtype=float)
    for i in range(h):
        b1 = np.copy(pre_boxes[i, :])
        b1[2:] = b1[:2] + b1[2:]
        for j in range(w):
            b2 = np.copy(next_boxes[j, :])
            b2[2:] = b2[:2] + b2[2:]
            delta_h = min(b1[2], b2[2]) - max(b1[0], b2[0])
            delta_w = min(b1[3], b2[3]) - max(b1[1], b2[1])
            if delta_h < 0 or delta_w < 0:
                expand_area = (max(b1[2], b2[2]) - min(b1[0], b2[0])) * (max(b1[3], b2[3]) - min(b1[1], b2[1]))
                area = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou[i, j] = -(expand_area - area) / area
            else:
                overlap = delta_h * delta_w
                area = (b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - max(overlap, 0)
                iou[i, j] = overlap / area

    return iou


class FeatureRecorder(object):
    """Historical features, boxes, sim, iou and etc
    """
    def __init__(self, max_record_frame=26, decay=1.0):
        """
        Parameters
        ----------
        max_record_frame: max record length of  features and bboxes
        decay: decay ratio of historical feature, sim(i-n,i)=sim(i-n,i)*decay^n
        """
        self.max_record_frame = max_record_frame
        self.decay = decay
        self.all_frame_index = np.array([], dtype=int)  # historical frame indexes
        self.all_features = {}                          # historical features
        self.all_boxes = {}                             # historical boxes
        self.all_similarity = {}                        # historical similarity between each two frames
        self.all_iou = {}                               # historical similarity between each two frames

    def update(self, dan, frame_index, features, boxes):
        """ Update record information
        Parameters
        ----------
        dan : DAN module
        frame_index: current frame index
        features: current features
        boxes: current boxes
        """
        # if the coming frame in the new frame
        # compute the sim and iou between historical frame bboxes and current frame bboxes
        if frame_index not in self.all_frame_index:
            # if the recorder have reached the max_record_frame.
            if len(self.all_frame_index) == self.max_record_frame:
                del_frame = self.all_frame_index[0]
                del self.all_features[del_frame]
                del self.all_boxes[del_frame]
                del self.all_similarity[del_frame]
                del self.all_iou[del_frame]
                self.all_frame_index = self.all_frame_index[1:]

            # add new item for all_frame_index, all_features and all_boxes. Besides, also add new similarity
            self.all_frame_index = np.append(self.all_frame_index, frame_index)
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes

            self.all_similarity[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:
                delta = pow(self.decay, frame_index - pre_index)
                pre_similarity = dan.forward_stacker_features(self.all_features[pre_index],
                                                              features, fill_up_column=False)
                self.all_similarity[frame_index][pre_index] = pre_similarity*delta

            self.all_iou[frame_index] = {}
            for pre_index in self.all_frame_index[:-1]:
                self.all_iou[frame_index][pre_index] = get_iou(self.all_boxes[pre_index], boxes)
        else:
            # compute the sim and iou between historical frame j bboxes(j < i) and frame i bboxes
            self.all_features[frame_index] = features
            self.all_boxes[frame_index] = boxes
            index = np.argwhere(self.all_frame_index == frame_index).item()

            for pre_index in self.all_frame_index[:index+1]:
                if pre_index == self.all_frame_index[-1]:
                    continue

                pre_similarity = dan.forward_stacker_features(self.all_features[pre_index], self.all_features[-1])
                self.all_similarity[frame_index][pre_index] = pre_similarity

                self.all_similarity[frame_index][pre_index] = get_iou(self.all_boxes[pre_index], boxes)

    def get_feature(self, frame_index, detection_index):
        """Get the feature by the specified frame index and detection index
        Note: feature: [N,C]
        Parameters
        ----------
        frame_index: index of frames
        detection_index: index of detection boxes, features[i,:]

        Returns
        -------
        The corresponding feature at frame index and detection index
        """

        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
            if len(features) == 0:
                return None
            if detection_index < len(features):
                return features[detection_index]

        return None

    def get_box(self, frame_index, detection_index):
        """Get the boxes by the specified frame index and detection index
        """
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
            if len(boxes) == 0:
                return None

            if detection_index < len(boxes):
                return boxes[detection_index]
        return None

    def get_features(self, frame_index):
        """Get the features([N,C]) by the specified frame index
        """
        if frame_index in self.all_frame_index:
            features = self.all_features[frame_index]
        else:
            return None
        if len(features) == 0:
            return None
        return features

    def get_boxes(self, frame_index):
        """Get the boxes by the specified frame index
        """
        if frame_index in self.all_frame_index:
            boxes = self.all_boxes[frame_index]
        else:
            return None

        if len(boxes) == 0:
            return None
        return boxes


class Node(object):
    """
    The Node is the basic element of a track. it contains the following information:
    1) extracted feature (it'll get removed when it isn't active)
    2) box  (l, t, r, b)
    3) label (active label indicating keeping the features)
    4) detection, the formated box
    """

    def __init__(self, frame_index, id, max_record_frame, max_track_node):
        self.frame_index = frame_index
        self.id = id  # detection index
        self.max_record_frame = max_record_frame
        self.max_track_node = max_track_node

    def get_box(self, frame_index, recoder):
        """Get the box of the trajectory in frame index
        """
        if frame_index - self.frame_index >= self.max_record_frame:
            return None
        return recoder.all_boxes[self.frame_index][self.id, :]

    def get_iou(self, frame_index, recoder, box_id):
        """Get the iou of two targets between target frame and origin node's frame
        """
        if frame_index - self.frame_index >= self.max_track_node:
            return None
        return recoder.all_iou[frame_index][self.frame_index][self.id, box_id]


class Track(object):
    _id_pool = 0

    def __init__(self, cfg):
        self.cfg = cfg
        self.nodes = list()  # trajectories
        self.id = Track._id_pool
        Track._id_pool += 1
        self.age = 0         # gaps since trajectory lost
        self.valid = True  # indicate this track is merged
        self.color = tuple((np.random.rand(3) * 255).astype(int).tolist())

    def __del__(self):
        for n in self.nodes:
            del n

    def add_age(self):
        self.age += 1

    def reset_age(self):
        self.age = 0

    def add_node(self, frame_index, recorder, node):
        """Add new node to trajectory, if the trajectory has already been lost,
           then will judge the iou, if matched, the age will be reset
        """
        # iou judge
        if len(self.nodes) > 0:
            n = self.nodes[-1]
            iou = n.get_iou(frame_index, recorder, node.id)
            delta_frame = frame_index - n.frame_index
            if delta_frame in self.cfg.tracker.min_iou_frame_gap:
                iou_index = self.cfg.tracker.min_iou_frame_gap.index(delta_frame)
                if iou < self.cfg.tracker.min_iou[iou_index]:
                    return False
        self.nodes.append(node)
        self.reset_age()
        return True

    def get_similarity(self, frame_index, recorder):
        """Sum up the similarity between target trajectory within max_track_node frames and all nodes in this frame
        """
        similarity = []
        for n in self.nodes:
            f = n.frame_index
            id = n.id
            if frame_index - f >= self.cfg.track.max_track_node:
                continue
            similarity += [recorder.all_similarity[frame_index][f][id, :]]

        if len(similarity) == 0:
            return None
        return np.sum(np.array(similarity), axis=0)

    def verify(self, frame_index, recorder, box_id):
        """Check the recorded iou in trajectories
        """
        for n in self.nodes:
            delta_f = frame_index - n.frame_index
            if delta_f in self.cfg.track.min_iou_frame_gap:
                iou_index = self.cfg.track.min_iou_frame_gap.index(delta_f)
                iou = n.get_iou(frame_index, recorder, box_id)
                if iou is None:
                    continue
                if iou < self.cfg.track.min_iou[iou_index]:
                    return False
        return True


class Tracks:
    """
    Track set. It contains all the tracks and manage the tracks. it has the following information
    1) tracks. the set of tracks
    2) keep the previous image and features
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = list()  # the set of tracks
        self.max_drawing_track = self.cfg.track.max_draw_track_node

    def __getitem__(self, item):
        return self.tracks[item]

    def append(self, track):
        """Add new track
        """
        self.tracks.append(track)
        self.volatile_tracks()

    def volatile_tracks(self):
        """if the number of current tracks > max objects, then delete the oldest track
        """
        if len(self.tracks) > self.cfg.track.max_object:
            # start to delete the most oldest tracks
            all_ages = [t.age for t in self.tracks]
            oldest_track_index = np.argmax(all_ages)
            del self.tracks[oldest_track_index]

    def get_track_by_id(self, id):
        for t in self.tracks:
            if t.id == id:
                return t
        return None

    def get_similarity(self, frame_index, recorder):
        """Sum up the similarity between all target trajectories within max_track_node frames and all nodes in this frame
            the last node is virtual node
            n x（m+1）-> n x (mxm+n): repeat each node m times and virtual node n times
        """
        ids = []
        similarity = []
        for t in self.tracks:
            s = t.get_similarity(frame_index, recorder)
            if s is None:
                continue
            similarity += [s]
            ids += [t.id]

        similarity = np.array(similarity)

        track_num = similarity.shape[0]
        if track_num > 0:
            box_num = similarity.shape[1]
        else:
            box_num = 0

        if track_num == 0:
            return np.array(similarity), np.array(ids)

        similarity = np.repeat(similarity, [1]*(box_num-1)+[track_num], axis=1)
        return np.array(similarity), np.array(ids)

    def one_frame_pass(self):
        """if skip one frame, then each track will be regarded as lost state, so the age+=1
        """
        keep_track_set = list()
        for i, t in enumerate(self.tracks):
            t.add_age()
            if t.age > self.cfg.track.max_track_age:
                continue
            keep_track_set.append(i)

        self.tracks = [self.tracks[i] for i in keep_track_set]

    def get_node_similarity(self, n1, n2, frame_index, recorder):
        """Get the similarity between two nodes
        """
        if n1.frame_index > n2.frame_index:
            n_max = n1
            n_min = n2
        elif n1.frame_index < n2.frame_index:
            n_max = n2
            n_min = n1
        else:  # in the same frame_index
            return None

        f_max = n_max.frame_index
        f_min = n_min.frame_index

        # not recorded in recorder
        if frame_index - f_min >= self.cfg.track.max_track_node:
            return None

        return recorder.all_similarity[f_max][f_min][n_min.id, n_max.id]

    def get_merge_similarity(self, t1, t2, frame_index, recorder):
        """Get the similarity between two tracks
        """
        merge_value = []
        if t1 is t2:
            return None

        all_f1 = [n.frame_index for n in t1.nodes]
        all_f2 = [n.frame_index for n in t2.nodes]

        for i, f1 in enumerate(all_f1):
            for j, f2 in enumerate(all_f2):
                compare_f = [f1 + 1, f1 - 1]
                for f in compare_f:
                    if f not in all_f1 and f == f2:
                        n1 = t1.nodes[i]
                        n2 = t2.nodes[j]
                        s = self.get_node_similarity(n1, n2, frame_index, recorder)
                        if s is None:
                            continue
                        merge_value += [s]

        if len(merge_value) == 0:
            return None
        return np.mean(np.array(merge_value))

    def core_merge(self, t1, t2):
        """Merge t2 to t1, after that t2 is set invalid
        """
        all_f1 = [n.frame_index for n in t1.nodes]
        all_f2 = [n.frame_index for n in t2.nodes]

        for i, f2 in enumerate(all_f2):
            if f2 not in all_f1:
                insert_pos = 0
                for j, f1 in enumerate(all_f1):
                    if f2 < f1:
                        break
                    insert_pos += 1
                t1.nodes.insert(insert_pos, t2.nodes[i])

        # remove some nodes in t1 in order to keep t1 satisfy the max nodes
        if len(t1.nodes) > self.cfg.track.max_track_node:
            t1.nodes = t1.nodes[-self.cfg.track.max_track_node:]
        t1.age = min(t1.age, t2.age)
        t2.valid = False

    def merge(self, frame_index, recorder):
        """Merge tracks if the track pair's mean similarity > min_merge_threshold
        """
        t_l = len(self.tracks)
        res = np.zeros((t_l, t_l), dtype=float)
        # get track similarity matrix
        for i, t1 in enumerate(self.tracks):
            for j, t2 in enumerate(self.tracks):
                s = self.get_merge_similarity(t1, t2, frame_index, recorder)
                if s is None:
                    res[i, j] = 0
                else:
                    res[i, j] = s

        # get the track pair which needs merged
        used_indexes = []
        merge_pair = []
        for i, t1 in enumerate(self.tracks):
            if i in used_indexes:
                continue
            max_track_index = np.argmax(res[i, :])
            if i != max_track_index and res[i, max_track_index] > self.cfg.track.min_merge_threshold:
                used_indexes += [max_track_index]
                merge_pair += [(i, max_track_index)]

        # start merge
        for i, j in merge_pair:
            self.core_merge(self.tracks[i], self.tracks[j])

        # remove the invalid tracks
        self.tracks = [t for t in self.tracks if t.valid]

    def show(self, frame_index, recorder, image):
        h, w, _ = image.shape

        # draw rectangle
        for t in self.tracks:
            if len(t.nodes) > 0 and t.age < 2:
                b = t.nodes[-1].get_box(frame_index, recorder)
                if b is None:
                    continue
                txt = '({}, {})'.format(t.id, t.nodes[-1].id)
                image = cv2.putText(image, txt, (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, t.color, 3)
                image = cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[0]+b[2]), int(b[1]+b[3])), t.color, 2)

        # draw line
        for t in self.tracks:
            if t.age > 1:
                continue
            if len(t.nodes) > self.max_drawing_track:
                start = len(t.nodes) - self.max_drawing_track
            else:
                start = 0
            for n1, n2 in zip(t.nodes[start:], t.nodes[start+1:]):
                b1 = n1.get_box(frame_index, recorder)
                b2 = n2.get_box(frame_index, recorder)
                if b1 is None or b2 is None:
                    continue
                c1 = (int(b1[0] + b1[2]/2.0), int(b1[1] + b1[3]))
                c2 = (int(b2[0] + b2[2] / 2.0), int(b2[1] + b2[3]))
                image = cv2.line(image, c1, c2, t.color, 2)

        return image


class DANTracker(object):
    def __init__(self, cfg):
        Track._id_pool = 0
        self.max_track_node = cfg.track.max_track_node
        self.max_track_age = cfg.track.max_track_age
        self.min_iou_frame_gap = cfg.track.min_iou_frame_gap
        self.min_iou = cfg.track.min_iou
        self.min_merge_threshold = cfg.track.min_merge_threshold
        self.max_bad_node = cfg.track.max_track_node
        self.max_drawing_track = cfg.track.max_draw_track_node
        self.cfg = cfg
        self.image_size = cfg.datasets.image_size
        self.dan = None
        self.build_model(cfg)

        self.frame_index = 0
        self.recorder = FeatureRecorder()
        self.tracks = Tracks(self.cfg)

    def build_model(self, cfg):
        print('=> Loading Model...')
        set_random_seed(26, deterministic=False, benchmark=True)
        devices = cfg.solver.device.split(',')
        self.cfg.solver.device = []
        for device in devices:
            device_id = int(device)
            if device_id >= 0:
                self.cfg.solver.device.append(torch.device(device_id))
            else:
                self.cfg.solver.device.append(torch.device('cpu'))

        self.dan = build_dan(self.cfg).to(self.cfg.solver.device[0])
        checkpoint = torch.load(self.cfg.io.resume, map_location=self.cfg.solver.device[0])
        self.dan.load_state_dict(checkpoint['state_dict'])
        self.dan.eval()

        print('=> Warming up Devices...')
        src_image = torch.rand(1, 3, self.image_size, self.image_size, device=self.cfg.solver.device[0])
        dst_image = torch.rand(1, 3, self.image_size, self.image_size, device=self.cfg.solver.device[0])
        src_centers = torch.rand(1, self.cfg.datasets.max_object, 1, 1, 2, device=self.cfg.solver.device[0])*2-1
        dst_centers = torch.rand(1, self.cfg.datasets.max_object, 1, 1, 2, device=self.cfg.solver.device[0])*2-1
        self.dan.get_similarity(src_image, src_centers, dst_image, dst_centers)

    def reset(self):
        self.frame_index = 0
        self.recorder = FeatureRecorder()
        self.tracks = Tracks(self.cfg)

    def update(self, image, detection, input_image, input_detection, frame_index, draw_image=False):
        """Get all the detection features and update the feature recorder

        Parameters
        ----------
        image: original opencv image
        detection: original (x,y,w,h) ndarray
        input_image: transformed image, BCHW
        input_detection: transformed detections, BN112
        frame_index: current frame index, if no detections in some frames, the update process will be skipped
        draw_image: whether to draw the trajectory
        """

        self.frame_index = frame_index
        features = self.dan.forward_feature_extracter(input_image, input_detection)
        self.recorder.update(self.dan, frame_index, features, detection.copy())

        if self.frame_index == 0 or len(self.tracks.tracks) == 0:
            for i in range(input_detection.shape[1]):
                t = Track(self.cfg)
                n = Node(self.frame_index, i,
                         self.cfg.track.max_record_frame,
                         self.cfg.track.max_track_node)
                t.add_node(self.frame_index, self.recorder, n)
                self.tracks.append(t)
            self.tracks.one_frame_pass()
            return self.tracks.show(self.frame_index, self.recorder, image)

        # get tracks similarity
        sim, ids = self.tracks.get_similarity(self.frame_index, self.recorder)

        if len(sim) > 0:
            # find the corresponding by the similar matrix
            row_index, col_index, _, _, _ = LinearAssignment(-sim)
            col_index[col_index >= detection.shape[0]] = -1

            # verification by iou
            verify_iteration = 0
            while verify_iteration < self.cfg.track.roi_verify_max_iteration:
                is_change = False
                for i in row_index:
                    box_id = col_index[i]
                    track_id = ids[i]

                    if box_id < 0:
                        continue
                    t = self.tracks.get_track_by_id(track_id)
                    if not t.verify(self.frame_index, self.recorder, box_id):
                        sim[i, box_id] *= self.cfg.track.roi_verify_punish_rate
                        is_change = True
                if is_change:
                    row_index, col_index, _, _, _ = LinearAssignment(-sim)
                    col_index[col_index >= detection.shape[0]] = -1
                else:
                    break
                verify_iteration += 1

            # update the tracks
            for i in row_index:
                track_id = ids[i]
                t = self.tracks.get_track_by_id(track_id)
                col_id = col_index[i]
                if col_id < 0:
                    continue
                node = Node(self.frame_index, col_id,
                            self.cfg.track.max_record_frame,
                            self.cfg.track.max_track_node)
                t.add_node(self.frame_index, self.recorder, node)

            # add new track
            for col in range(len(detection)):
                if col not in col_index:
                    node = Node(self.frame_index, col,
                                self.cfg.track.max_record_frame,
                                self.cfg.track.max_track_node)
                    t = Track(self.cfg)
                    t.add_node(self.frame_index, self.recorder, node)
                    self.tracks.append(t)

        # remove the old track
        self.tracks.one_frame_pass()

        if draw_image:
            image = self.tracks.show(self.frame_index, self.recorder, image)

        return image





