# -*- coding: utf-8 -*-
# Author : hongweiwang
# Email  : hongweiwang@hust.edu.cn
# Date   : 26/11/2019

import numpy as np
import colorsys
import cv2
import time
import os

DEFAULT_UPDATE_MS = 20


class ImageViewer(object):
    """An image viewer with drawing routines and video capture capabilities.

    Key Bindings:

    * 'SPACE' : pause
    * 'ESC' : quit
    * 's' : step

    Parameters
    ----------
    update_ms : int
        Number of milliseconds between frames (1000 / frames per second).
    window_shape : (int, int)
        Shape of the window (width, height).
    caption : Optional[str]
        Title of the window.

    Attributes
    ----------
    image : ndarray
        Color image of shape (height, width, 3). You may directly manipulate
        this image to change the view. Otherwise, you may call any of the
        drawing routines of this class. Internally, the image is treated as
        beeing in BGR color space.

        Note that the image is resized to the the image viewers window_shape
        just prior to visualization. Therefore, you may pass differently sized
        images and call drawing routines with the appropriate, original point
        coordinates.
    color : (int, int, int)
        Current BGR color code that applies to all drawing routines.
        Values are in range [0-255].
    text_color : (int, int, int)
        Current BGR text color code that applies to all text rendering
        routines. Values are in range [0-255].
    thickness : int
        Stroke width in pixels that applies to all drawing routines.

    """

    def __init__(self, update_ms, window_shape=(640, 480), caption="Figure 1"):
        self._window_shape = window_shape
        self._caption = caption
        self._update_ms = update_ms
        self._video_writer = None
        self._user_fun = lambda: None
        self._terminate = False

        self.image = np.zeros(self._window_shape + (3,), dtype=np.uint8)
        self._color = (0, 0, 0)
        self.text_color = (255, 255, 255)
        self.thickness = 1

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        if len(value) != 3:
            raise ValueError("color must be tuple of 3")
        self._color = tuple(int(c) for c in value)

    def rectangle(self, x, y, w, h, label=None):
        """Draw a rectangle.

        Parameters
        ----------
        x : float | int
            Top left corner of the rectangle (x-axis).
        y : float | int
            Top let corner of the rectangle (y-axis).
        w : float | int
            Width of the rectangle.
        h : float | int
            Height of the rectangle.
        label : Optional[str]
            A text label that is placed at the top left corner of the
            rectangle.

        """
        pt1 = int(x), int(y)
        pt2 = int(x + w), int(y + h)
        cv2.rectangle(self.image, pt1, pt2, self._color, self.thickness)
        if label is not None:
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_PLAIN, 1, self.thickness)

            center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
                  text_size[0][1]
            cv2.rectangle(self.image, pt1, pt2, self._color, -1)
            cv2.putText(self.image, label, center, cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 255, 255), self.thickness)

    def rectangle_tail(self, x, y, w, h):
        pt3 = int(x + w / 2 - w / 12), int(y + h - w / 12)
        pt4 = int(x + w / 2 + w / 12), int(y + h + w / 12)
        cv2.rectangle(self.image, pt3, pt4, self._color, -1)

    def enable_videowriter(self, output_filename, fourcc_string="MJPG",
                           fps=None):
        """ Write images to video file.

        Parameters
        ----------
        output_filename : str
            Output filename.
        fourcc_string : str
            The OpenCV FOURCC code that defines the video codec (check OpenCV
            documentation for more information).
        fps : Optional[float]
            Frames per second. If None, configured according to current
            parameters.

        """
        fourcc = cv2.VideoWriter_fourcc(*fourcc_string)
        if fps is None:
            fps = int(1000. / self._update_ms)
        self._video_writer = cv2.VideoWriter(
            output_filename, fourcc, fps, self._window_shape)

    def run(self, update_fun=None):
        """Start the image viewer.

        This method blocks until the user requests to close the window.

        Parameters
        ----------
        update_fun : Optional[Callable[] -> None]
            An optional callable that is invoked at each frame. May be used
            to play an animation/a video sequence.

        """
        if update_fun is not None:
            self._user_fun = update_fun

        self._terminate, is_paused = False, False
        # print("ImageViewer is paused, press space to start.")
        while not self._terminate:
            t0 = time.time()
            if not is_paused:
                self._terminate = not self._user_fun()
                if self._video_writer is not None:
                    self._video_writer.write(
                        cv2.resize(self.image, self._window_shape))
            t1 = time.time()
            remaining_time = max(1, int(self._update_ms - 1e3 * (t1 - t0)))
            cv2.imshow(
                self._caption, cv2.resize(self.image, self._window_shape[:2]))
            key = cv2.waitKey(remaining_time)
            if key & 255 == 27:  # ESC
                print("terminating")
                self._terminate = True
            elif key & 255 == 32:  # ' '
                print("toggeling pause: " + str(not is_paused))
                is_paused = not is_paused
            elif key & 255 == 115:  # 's'
                print("stepping")
                self._terminate = not self._user_fun()
                is_paused = True

        # Due to a bug in OpenCV we must call imshow after destroying the
        # window. This will make the window appear again as soon as waitKey
        # is called.
        #
        # see https://github.com/Itseez/opencv/issues/4535
        self.image[:] = 0
        cv2.destroyWindow(self._caption)
        cv2.waitKey(1)
        cv2.imshow(self._caption, self.image)


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms):
        image_shape = seq_info["image_size"][::-1]
        #aspect_ratio = float(image_shape[1]) / image_shape[0]
        #image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_tail(self, track_ids, frame_id, results):
        self.viewer.thickness = 2
        for track_id in (track_ids):
            boolean = []
            for i in range(len(results)):
                boolean.append(results[i, 1].astype(np.int) == track_id and \
                               results[i, 0].astype(np.int) >= (frame_id - 9) and \
                               results[i, 0].astype(np.int) <= frame_id)
            boolean = np.array(boolean).astype(np.bool_)

            self.viewer.color = create_unique_color_uchar(track_id)
            boxes_tails = results[boolean, 2:6]
            for box_tail in (boxes_tails):
                self.viewer.rectangle_tail(*box_tail.astype(np.int))


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """

    image_dir = os.path.join(sequence_dir, "img1")

    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    # groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    # if os.path.exists(groundtruth_file):
    #   groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def run(sequence_dir, result_file, show_false_alarms=False, detection_file=None,
        update_ms=None, video_filename=None):
    """Run tracking result visualization.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    result_file : str
        Path to the tracking output file in MOTChallenge ground truth format.
    show_false_alarms : Optional[bool]
        If True, false alarms are highlighted as red boxes.
    detection_file : Optional[str]
        Path to the detection file.
    update_ms : Optional[int]
        Number of milliseconds between cosecutive frames. Defaults to (a) the
        frame rate specifid in the seqinfo.ini file or DEFAULT_UDPATE_MS ms if
        seqinfo.ini is not available.
    video_filename : Optional[Str]
        If not None, a video of the tracking results is written to this file.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    if os.path.isfile(result_file):
        results = np.loadtxt(result_file, delimiter=',')
    else:
        results = result_file
        seq_info['max_frame_idx'] = max(results[:, 0])

    def frame_callback(vis, frame_idx):
        print("Frame idx", frame_idx)
        image = cv2.imread(
            seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

        vis.set_image(image.copy())

        mask = results[:, 0].astype(np.int) == frame_idx
        track_ids = results[mask, 1].astype(np.int)
        boxes = results[mask, 2:6]
        vis.draw_groundtruth(track_ids, boxes)
        vis.draw_tail(track_ids, frame_idx, results)

    if update_ms is None:
        update_ms = seq_info["update_ms"]
    if update_ms is None:
        update_ms = DEFAULT_UPDATE_MS
    visualizer = Visualization(seq_info, update_ms)
    if video_filename is not None:
        visualizer.viewer.enable_videowriter(video_filename)
    visualizer.run(frame_callback)


def convert(filename_in, filename_out, ffmpeg_executable="ffmpeg"):
    """
    convert the videos from .avi to .mp4
    """
    import subprocess
    command = [ffmpeg_executable, "-i", filename_in, "-c:v", "libx264",
               "-preset", "slow", "-crf", "21", filename_out]
    subprocess.call(command)


def generate_videos(mot_dir, result_dir, output_dir=None, convert_h264=False, update_ms=None):
    """
    generate videos of the tracking results

    Parameters
    ----------
    mot_dir: "Path to MOTChallenge directory (train or test)"
    result_dir: "Path to the folder with tracking output. or just file path"
    output_dir: "Folder to store the videos in. Will be created if it does not exist."
    convert_h264: "If true, convert videos to libx264 (requires FFMPEG)"
    update_ms: "Time between consecutive frames in milliseconds. "
        "Defaults to the frame_rate specified in seqinfo.ini, if available."
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = './'
    video_filename = None

    if os.path.isdir(result_dir):
        result_list = os.listdir(result_dir)
    elif os.path.isfile(result_dir):
        result_list = [result_dir]
    else:
        if '/' in mot_dir:
            sequence = mot_dir.split('/')
        else:
            sequence = mot_dir.split('\\')
        for s in reversed(sequence):
            if len(s) > 0:
                sequence = s
                break
        video_filename = os.path.join(output_dir, "%s.avi" % sequence)
        run(mot_dir, result_dir, False, None, update_ms, video_filename)
        return

    for sequence_txt in result_list:
        sequence = os.path.splitext(sequence_txt)[0]
        sequence = sequence.split('/')[-1]
        sequence = sequence.split('\\')[-1]
        print(mot_dir)

        if not os.path.exists(mot_dir):
            continue

        video_filename = os.path.join(output_dir, "%s.avi" % sequence)
        print("Saving %s to %s." % (sequence_txt, video_filename))
        run(mot_dir, sequence_txt, False, None, update_ms, video_filename)

    if not convert_h264:
        #import sys
        #sys.exit()
        return
    for sequence_txt in result_list:
        sequence = os.path.splitext(sequence_txt)[0]
        sequence = sequence.split('/')[-1]
        sequence = sequence.split('\\')[-1]
        if not os.path.exists(mot_dir):
            continue
        filename_in = os.path.join(output_dir, "%s.avi" % sequence)
        filename_out = os.path.join(output_dir, "%s.mp4" % sequence)
        convert(filename_in, filename_out)


if __name__ == '__main__':

    generate_videos(mot_dir=r'E:\\datasets\\MOT17\\test\\MOT17-01-SDP',
                    result_dir=r'C:\\Users\\Hasee\\Desktop\\毕设\\IJCAI\ablation\\MOT17_motion_reid\\MOT17-01-SDP.txt',
                    output_dir=r'E:\\datasets\\video\\',
                    convert_h264 = False, update_ms = None)
