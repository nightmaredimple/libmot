# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 26/4/2020

from libmot.tools import Config
from libmot.tracker.DAN import DANAugmentation, DANTracker
from libmot.utils import DataLoader, mkdir_or_exist, check_folder_exist, evaluation_mot
from libmot.datasets import MOTReader
import cv2
import numpy as np
import os
import argparse
import time
from tqdm._tqdm import trange
import torch

parser = argparse.ArgumentParser(description='DAN Tracker')
parser.add_argument('--config_file', type=str, default='examples/DAN/track.yaml', help='config path')
parser.add_argument('--sequence_folder', default=None, help='single sequence folder, if not defined, '
                                                            'all sequences under mot root with phase will be tracked')
parser.add_argument('--phase', default=None, help='phase can be train/test/None')
parser.add_argument('--result_folder', type=str, default='./results', help='video saving or result saving folder')
parser.add_argument('--show_image', action='store_true', help='show image if true, or hidden')
parser.add_argument('--save_video', action='store_true', help='save video if true')
parser.add_argument('--evaluate', action='store_true', help='if defined, the results will be evaluated')

args = parser.parse_args()


@torch.no_grad()
def test(tracker, cfg):

    def transform(data):
        dets = mot_reader.detection['bboxes'][data['index']]
        image = data['raw']

        if dets.size == 0:
            return image, dets, None, None

        if len(dets) > cfg.datasets.max_object:
            dets = dets[:cfg.datasets.max_object, :]
        input_dets = dets.copy()
        input_dets[:, 2:] += input_dets[:, :2]

        input_image = image.copy()
        input_image, _, input_dets, _, _ = augmentation(img_pre=input_image, boxes_pre=input_dets)
        input_image = input_image.unsqueeze(0).to(cfg.solver.device[0])
        input_dets = input_dets.unsqueeze(0).to(cfg.solver.device[0])
        return image, dets, input_image, input_dets

    mot_reader = MOTReader(args.sequence_folder, vis_thresh=cfg.datasets.min_visibility,
                           detection_thresh=cfg.datasets.detection_thresh)
    sequence_name = mot_reader.video_info['name']
    w, h = mot_reader.video_info['shape']
    print('=> Preparing DataLoader for {}...'.format(sequence_name))
    tracker.reset()
    augmentation = DANAugmentation(cfg, type='test')
    method = {'transform': transform}
    loader = DataLoader(os.path.join(args.sequence_folder,'img1'), max_size=20, **method)
    loader.start()
    time.sleep(2)

    result_file = os.path.join(args.result_folder, sequence_name, '.txt')
    result_video = os.path.join(args.result_folder, sequence_name + '.avi')
    if args.save_video:
        video_writer = cv2.VideoWriter(result_video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (w, h))

    print('=> Begin to Track {}...'.format(sequence_name))
    result = []
    for i in trange(len(mot_reader)):
        image, dets, input_image, input_dets = loader.getData()['output']
        if dets.size == 0:
            continue

        draw = tracker.update(image, dets, input_image, input_dets, i, args.show_image)
        if args.show_image and not image_org is None:
            cv2.imshow(sequence_name, draw)
            cv2.waitKey(1)

        if args.save_video:
            video_writer.write(draw)

        for t in tracker.tracks:
            n = t.nodes[-1]
            if t.age == 1:
                b = n.get_box(tracker.frame_index - 1, tracker.recorder)
                result.append(
                    [i] + [t.id] + [b[0], b[1], b[2], b[3]] + [-1, -1, -1, -1]
                )

    np.savetxt(result_file, np.array(result).astype(int), fmt='%i')
    if args.evaluate and mot_reader.ground_truth is not None:
        mota, _, idf1, idsw = evaluation_mot(mot_reader.ground_truth, result)
        print('MOTA={:.4f}, IDF1={:.4f}, ID Sw.={}'.format(mota, idf1, idsw))

    loader.stop()


if __name__ == "__main__":
    cfg = Config.fromfile(args.config_file)
    cfg.io.resume = 'E:\\datasets\\log\\20200425_18_45_37\\checkpoints\\dan_best.pth.tar'  # model path
    cfg.solver.device = '-1'

    mkdir_or_exist(args.result_folder)
    dan = DANTracker(cfg)
    if args.sequence_folder is None:
        check_folder_exist(cfg.io.mot_root)
        if args.phase is None:
            args.phase = ['train', 'test']
        else:
            args.phase = [args.phase]
        for phase in os.listdir(cfg.io.mot_root):
            if phase not in args.phase:
                continue
            for sequence in os.listdir(cfg.io.mot_root + phase):
                current_sequence = os.path.join(cfg.io.mot_root, phase, sequence)
                args.sequence_folder = current_sequence
                test(dan, cfg)
    else:
        check_folder_exist(args.sequence_folder)
        test(dan, cfg)





