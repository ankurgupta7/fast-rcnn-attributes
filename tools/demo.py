#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = {'__background__':0, # always index 0
             'is_male':1,
             'has_long_hair':2,
             'has_glasses':3,
             'has_hat':4,
             'has_t-shirt':5,
             'has_long_sleeves':6,
             'has_shorts':7,
             'has_jeans':8,
             'has_long_pants':9}

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name, classes):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',
                            image_name + '_boxes.mat')
    obj_proposals = sio.loadmat(box_file)['boxes']
    obj_proposals = obj_proposals[0][0]
    # Load the demo image
    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name + '.jpg')
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.2
    NMS_THRESH = 0.001
    #for cls in classes:
    cls_ind_all = [CLASSES[x] for x in classes]
    # boxes_shape = boxes[:, 4*0:4*(0 + 1)].shape
    cls_boxes = boxes[:, 4*cls_ind_all[0]:4*(cls_ind_all[0] + 1)]
    # for cls_ind in cls_ind_all[1:]:
    #     cls_boxes =  np.hstack((cls_boxes,boxes[:, 4*cls_ind:4*(cls_ind + 1)]))
    cls_scores = scores[:, cls_ind_all]
    # keep = np.where(cls_scores >= CONF_THRESH)[0]
    keep = np.where((cls_scores >= CONF_THRESH).all(axis=1))[0]
    cls_boxes = cls_boxes[keep, :]
    cls_scores = cls_scores[keep,:]
    # dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)
    cls_scores_0 = cls_scores[:, 0]
    dets = np.hstack(( cls_boxes,
                       cls_scores_0[:,np.newaxis])).astype(np.float32)

    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    print 'All {} detections with p({} | box) >= {:.1f}'.format(classes, classes,
                                                                CONF_THRESH)
    vis_detections(im, classes, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    # / home / agupta82 / fast - rcnn / models / attributes / vgg / fast_rcnn / test.prototxt
    prototxt = os.path.join(cfg.ROOT_DIR, 'models','attributes','vgg',
                            'fast_rcnn', 'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output', 'default',
                              'train','vgg_cnn_m_1024_fast_rcnn_iter_100000.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/scripts/'
                       'fetch_fast_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Demo for data/demo/00003.jpg'
    demo(net, '00003', (
             'is_male',
             # 'has_long_hair',
             # 'has_glasses',
             # 'has_hat',
             #  'has_t-shirt',
              'has_long_sleeves',
             # 'has_shorts',
             #  'has_jeans',
             # 'has_long_pants',
                        ))

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    plt.show()
