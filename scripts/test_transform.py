from libmot.tracker.DAN import DANAugmentation
import cv2
import numpy as np

T = DANAugmentation(size=900, mean=(104, 117, 123), type='test', max_object=5,
                    max_expand=1.2, lower_contrast=0.7, upper_constrast=1.5,
                    lower_saturation=0.7, upper_saturation=1.5)
img_pre = cv2.imread('E:\\datasets\\MOT17\\train\\MOT17-04-SDP\\img1\\000001.jpg')
img_next = cv2.imread('E:\\datasets\\MOT17\\train\\MOT17-04-SDP\\img1\\000002.jpg')
boxes_pre = np.array([[20, 40, 100.0, 150.0], [50.0, 100.0, 400, 150.0]])
boxes_next = np.array([[20, 40, 100.0, 150.0]])
labels = np.zeros((5, 5))
labels[0, 0] = 1


T(img_pre=img_pre, img_next=img_next, boxes_pre=boxes_pre, boxes_next=boxes_next, labels=labels)
