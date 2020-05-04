# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 11/18/2019

import os
import cv2
import numpy as np
import threading
from queue import Queue
from copy import deepcopy
import time


class DataLoader(object):

    def __init__(self, image_path, image_list = None, max_size = 8, color_space = 'BGR', output_size = None, \
                 norm = False, norm_method = 'default', mean = None, std = None, image_type = None,
                 interpolation = cv2.INTER_LINEAR, save_list = ['output'], format = np.uint8, **kwargs):
        """Data Loader using producer-customer models

        Parameters
        -------------
        image_path: str
            directory for images' path
        image_list: str or List[image_names]
            image_list path or List[image_names], if not defined, will search the entir directory
        max_size: int
            max size of queue for prefetch
        color_space: str
            you can choose which colsor space to be converted,eg: 'BGR','RGB',HSV','Gray'
            default is BGR
        output_size: Tuple(width, height)
            the output image will be resized to the output_size
        norm: bool
            whether to normalize the image
        norm_method: str
            methods to normalize, default is default,you can choose:
            default, MINMAX、L1、L2、MeanStd
        mean: float or List[float]
            the mean value of each channel for normalized
        std: float or List[float]
            the std value of each channel for normalized
        image_type: List[type]
            eg:['jpg','png','JPEG']
        interpolation: cv2.method
            method of interpolation,eg: cv2.INTER_LINEAR, cv2.INTER_CUBIC
        save_list: List[str]
            List of variable will be putted into queue
            eg: [color,norm, resize, output, method0]
        format: dtype
            the output image wil be saved as this dtype
        **kwarg: Dict of self defined method for transformation {name : function}
            eg: {padding: lambda x: cv2.copyMakeBorder(x,0,0,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])
        """
        assert image_path is not None, "Please define the image directory!"
        assert isinstance(output_size, tuple) or output_size is None, "output size must be (width, height) tuple like"
        assert type(mean) == type(std), "mean must be same format as std"

        if os.path.isdir(image_path):
            self.image_path = image_path
        else:
            raise ValueError("image path must be directory")

        if image_type is None:
            self.image_type = ['jpg', 'png', 'JPEG', 'bmp']
        else:
            self.image_type = image_type if isinstance(image_type, list) else [image_type]

        if image_list is None:
            self.image_list =  sorted(os.listdir(self.image_path))
            self.image_list = [name for name in self.image_list if name.split('.')[-1] in self.image_type]

        elif isinstance(image_list, str) and os.path.isfile(image_list):
            self.image_list = []
            with open(image_list, 'r') as f:
                for line in f:
                    if line.strip('\n').split('.')[-1] in self.image_type:
                        self.image_list.append(line.strip('\n'))
        elif isinstance(image_list, list):
            self.image_list = [name for name in self.image_list if name.split('.')[-1] in self.image_type]
        else:
            raise ValueError("illegal image list")
        self.norm = norm
        self.norm_method = norm_method

        if mean is not None:
            self.mean = list(mean)
            self.std = list(std)
            if norm and norm_method == 'MeanStd':
                assert self.color_space == 'Gray' and len(self.mean) == 1,\
                    "self defined mean/std must be float when color space is gray"
                assert self.color_space is not 'Gray' and len(self.mean) > 1, \
                    "self defined mean/std must be a list when color space is not gray"
        else:
            self.mean = mean
            self.std = std

        self.color_space = color_space
        self.output_size = output_size
        self.interpolation = interpolation
        self.format = np.float32 if self.norm else format
        self.save_list = save_list

        self.method = kwargs

        self.index = 0
        self.len = len(self.image_list)
        self.max_size = int(max(max_size, 1))
        self.queue = Queue(self.max_size)
        self.left = deepcopy(self.len)

    def start(self):
        """start the thread

        """
        if self.len > 0:
            self.alive = True
            self.thread_read = threading.Thread(target=self.process)
            self.thread_read.setDaemon(True)
            self.thread_read.start()
            return True
        else:
            print('The image List is empty!')
            return False

    def process(self):
        """Read and Process images

        """
        while self.alive:
            if self.index >= self.len:
                if self.queue.empty():
                    break
                else:
                    time.sleep(0.05)
                    continue
            time0 = time.time()
            img = cv2.imread(os.path.join(self.image_path, self.image_list[self.index]))

            # convert to other color space
            if self.color_space == 'RGB':
                img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif self.color_space == 'HSV':
                img_color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif self.color_space == 'Gray':
                img_color = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_color = img

            # resize
            if self.output_size is not None:
                img_resize = cv2.resize(img_color, self.output_size, interpolation=self.interpolation)
            else:
                img_resize = img_color

            # normalization
            img_norm = img_resize.astype(np.float32)
            if self.norm:
                if self.norm_method == 'MINMAX':
                    img_norm = cv2.normalize(img_norm, None, 0, 1, cv2.NORM_MINMAX)
                elif self.norm_method == 'L1':
                    img_norm = cv2.normalize(img_norm, None, 1, 0, cv2.NORM_L1)
                elif self.norm_method == 'L2':
                    img_norm = cv2.normalize(img_color, None, 1, 0, cv2.NORM_L2)
                elif self.norm_method == 'MeanStd':
                    if self.mean is not None:
                        img_norm = (img_norm - self.mean) / self.std
                    else:
                        mean, std = cv2.meanStdDev(img_norm)
                        img_norm = (img_norm - mean.reshape(1,1,3)) / (std.reshape(1,1,3))
                elif self.norm_method == 'default':
                    img_norm = img_norm / 255.0

            # data
            data = {'name': self.image_list[self.index],
                    'raw': img,
                    'index': self.index}
            if self.color_space != 'BGR':
                data.update({'color': img_color})
            if self.output_size is not None:
                data.update({'resize': img_resize})
            if self.norm:
                data.update({'norm': img_norm})

            # other method
            img_output = img_norm.astype(self.format)

            for key, method in self.method.items():

                img_output = method(data)
                data.update({key: deepcopy(img_output)})

            q_data = {'name': self.image_list[self.index],
                      'index': self.index}
            for v in self.save_list:
                if 'raw' in v:
                    q_data.update({v: img})
                elif 'color' in v and self.color_space != 'BGR':
                    q_data.update({v: img_color})
                elif 'resize' in v and self.output_size is not None:
                    q_data.update({v: img_resize})
                elif 'norm' in v and self.norm:
                    q_data.update({v: img_norm})
                elif 'output' in v:
                    q_data.update({'output': img_output})
                elif v in self.method.keys():
                    q_data.update({v: data[v]})

            # queue
            self.queue.put(q_data)
            self.index += 1

    def getData(self):
        if self.alive and self.left > 0:
            try:
                self.left -= 1
                return self.queue.get(block = True, timeout = 10)
            except Exception:
                return None
        else:
            return None

    def stop(self):
        """Stop the thread

        """
        self.alive = False
        if self.thread_read:
            self.thread_read.join()


if __name__ == '__main__':
    import torch
    method = {'CUDATensor': lambda x: (torch.from_numpy(x['norm'])).cuda()}
    loader = DataLoader('images/MOT17-10-img', max_size=20, norm=True, \
                        color_space='RGB', save_list = ['raw','output'],**method)
    cost = 0
    count = 0
    if loader.start():
        time.sleep(10) # for data prefetch
        while loader.alive:
            time0 = time.time()
            data = loader.getData()
            if data is None:
                loader.stop()
                break
            else:
                img = data['output']

            cost += time.time() - time0
            count += 1
            time.sleep(0.04)
    else:
        print('DataLoader Error!')

    print('Imread + BGR2RGB + Normalization + ToTensor + ToCuda + costs avg: %.3f s' % (cost/count))



