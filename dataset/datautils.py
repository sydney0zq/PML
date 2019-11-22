#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qiang.zhou <theodoruszq@gmail.com>

"""
Dataset utils.
"""

import numpy as np
import cv2
import random


class Auger(object):
    """
        Simple augmentation abstract class:
        :cropsize: (h, w) crop region
        :mirror_aug: Whether include mirror operation
    """
    def __init__(self, cropsize, mirror_aug=True):
        self.cropsize = cropsize
        self.mirror_aug = mirror_aug

    def aug(self, image, label, min_iou=0.5):
        crop_h, crop_w = self.cropsize
        image, label = np.asarray(image, np.float32), np.asarray(label)
        image_bak, label_bak = image.copy(), label.copy()
        cnt = 0

        while True:
            img_h, img_w = label.shape
            pad_h, pad_w = max(crop_h-img_h, 0), max(crop_w-img_w, 0) 
            if pad_h > 0 or pad_w > 0:
                img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                    pad_w, cv2.BORDER_CONSTANT, 
                    value=(0.0, 0.0, 0.0))
                label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                    pad_w, cv2.BORDER_CONSTANT,
                    value=(0,))
            else:
                img_pad, label_pad = image, label
            img_h, img_w = label_pad.shape
            h_off = random.randint(0, img_h - crop_h)
            w_off = random.randint(0, img_w - crop_w)
            image = np.asarray(img_pad[h_off:h_off+crop_h, w_off:w_off+crop_w], np.float32)
            label = np.asarray(label_pad[h_off:h_off+crop_h, w_off:w_off+crop_w], np.float32)
            # As we use PIL library, we get RGB
            if self.mirror_aug:
                flip = np.random.choice(2) * 2 - 1
                image = image[:, ::flip, :]      # Mirror and RGB
                label = label[:, ::flip]

            if np.sum(label) > np.sum(label_bak)*min_iou or np.sum(label_bak) < 50 or cnt >= 5:
                break
            else:
                image, label = image_bak, label_bak

            cnt += 1

        return np.array(image), np.array(label)


def get_mask_bbox(m, border_pixels=0):
    if not np.any(m):
        return (0, 0, m.shape[1], m.shape[0])
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    h,w = m.shape
    ymin = max(0, ymin - border_pixels)
    ymax = min(h-1, ymax + border_pixels)
    xmin = max(0, xmin - border_pixels)
    xmax = min(w-1, xmax + border_pixels)
    return (xmin, ymin, xmax, ymax)

#cropimage = lambda x, bbox: np.array(Image.fromarray(x.astype(np.uint8)).crop(bbox), x.dtype)
""" image: ndarray/PIL; box: cross format """
def cropimage(image, box):
    image = np.array(image)
    minx, miny, maxx, maxy = [int(x) for x in box]
    width, height = max(0, maxx - minx), max(0, maxy - miny)
    assert (not(width == 0 or height == 0)), \
            "Nonsense cropbox: ({}, {}, {}, {})...".format(minx, miny, maxx, maxy)

    roi_left = max(0, min(minx, image.shape[1]))
    roi_right = min(image.shape[1], max(0, maxx))
    roi_top = max(0, min(miny, image.shape[0]))
    roi_bottom = min(image.shape[0], max(0, maxy))
    roi_width, roi_height = roi_right-roi_left, roi_bottom-roi_top
    pad_x, pad_y = max(0, 0-minx), max(0, 0-miny)
    
    if len(image.shape) == 3:
        padimage = np.zeros((height, width, 3))
        padimage[pad_y:pad_y+roi_height, pad_x:pad_x+roi_width, :] = image[roi_top:roi_bottom, roi_left:roi_right, :]
    elif len(image.shape) == 2:
        padimage = np.zeros((height, width))
        padimage[pad_y:pad_y+roi_height, pad_x:pad_x+roi_width] = image[roi_top:roi_bottom, roi_left:roi_right]

    return padimage

if __name__ == "__main__":
    x = np.zeros((800, 800, 3))
    x[100:200, 200:300, 0] = 128
    cropimage(x, [0, 0, 1, 1])
    exit()
