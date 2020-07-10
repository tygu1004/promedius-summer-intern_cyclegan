# -*- coding: utf-8 -*-
"""
Module:    input_util.py
Language:  Python3
Date:      2020-06-30 14:16:00
Version:   open.VER 1.0
Developer: LEE GAEUN  (ggelee93@gmail.com) / @author: yeohyeongyu
Copyright (c) Promedius.
All rights reserved.
"""

import os
from glob import glob
import tensorflow as tf
import numpy as np
import dicom
from datetime import datetime


# psnr
def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def tf_psnr(img1, img2, PIXEL_MAX=255.0):
    mse = tf.reduce_mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * log10(PIXEL_MAX / tf.sqrt(mse))


def TaskID_Generator():
    currentTime = datetime.now()
    strTime = "%04d%02d%02d_%02d%02d%02d" % (
        currentTime.year, currentTime.month, currentTime.day, currentTime.hour, currentTime.minute, currentTime.second)
    return strTime


def rescale_arr(data, i_min, i_max, o_min, o_max, out_dtype=None):
    if not out_dtype:
        out_dtype = data.dtype
    scale = float(o_max - o_min) / (i_max - i_min)
    out_data = (data - i_min) * scale + o_min
    return tf.dtypes.cast(out_data, out_dtype)


def load_scan(path):
    slices = [dicom.read_file(s) for s in path]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = \
            np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(slices, pre_fix_nm=''):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    image[image == -2000] = 0

    digit = 4
    slice_nm = []
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float32)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)

        # sorted(idx), sorted(d_idx)  -> [1, 10, 2], [ 0001, 0002, 0010]
        s_idx = str(slice_number)
        d_idx = '0' * (digit - len(s_idx)) + s_idx
        slice_nm.append(pre_fix_nm + '_' + d_idx)
    return np.array(image, dtype=np.int16), slice_nm


def normalize(img, max_=3071, min_=-1024):
    img = img.astype(np.float32)
    img = (img - min_) / (max_ - min_)
    return img


class DCMDataLoader(object):
    def __init__(self, dcm_path, image_size=512, patch_size=64, image_max=3071,
                 image_min=-1024, batch_size=1, extension='dcm'):
        # dicom file dir
        self.extension = extension
        self.dcm_path = dcm_path

        # image params
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_max = image_max
        self.image_min = image_min

        # training params
        self.batch_size = batch_size

        # CT slice name
        self.LDCT_image_name, self.NDCT_image_name = [], []

    # dicom file -> numpy array
    def __call__(self, patent_no_list_A, patent_no_list_B):
        def get_images(patent_no_list, domain_name):
            p = []
            image_name = []
            for patent_no in patent_no_list:
                p_path = glob(os.path.join(self.dcm_path, patent_no, '*.' + self.extension))
                # load images
                org_images, slice_nm = get_pixels_hu(load_scan(p_path),
                                                     '{}_{}'.format(patent_no, domain_name))

                # CT slice name
                image_name.extend(slice_nm)

                # normalization
                p.append(normalize(org_images, self.image_max, self.image_min))

            return p, image_name

        p_LDCT, self.LDCT_image_name = get_images(patent_no_list_A, 'A')
        p_NDCT, self.NDCT_image_name = get_images(patent_no_list_B, 'B')

        self.LDCT_images = np.concatenate(tuple(p_LDCT), axis=0)
        self.NDCT_images = np.concatenate(tuple(p_NDCT), axis=0)

        # image index
        self.LDCT_index, self.NDCT_index = list(range(len(self.LDCT_images))), list(range(len(self.NDCT_images)))

    def get_train_set(self, patch_size, whole_size=512):
        whole_h = whole_w = whole_size
        h = w = patch_size

        # patch image range
        hd, hu = h // 2, int(whole_h - np.round(h / 2))
        wd, wu = w // 2, int(whole_w - np.round(w / 2))

        # patch image center(coordinate on whole image)
        h_pc, w_pc = np.random.choice(range(hd, hu + 1)), np.random.choice(range(wd, wu + 1))

        ldct_patch_set = tf.data.Dataset.from_tensor_slices(tf.expand_dims(self.LDCT_images, axis=-1))
        ndct_patch_set = tf.data.Dataset.from_tensor_slices(tf.expand_dims(self.NDCT_images, axis=-1))

        ldct_patch_set = ldct_patch_set.map(
            lambda x: x[h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))])
        ndct_patch_set = ndct_patch_set.map(
            lambda x: x[h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))])

        ldct_patch_set = ldct_patch_set.batch(self.batch_size)
        ndct_patch_set = ndct_patch_set.batch(self.batch_size)

        return ldct_patch_set, ndct_patch_set

    def get_test_set(self):
        ldct_set = tf.data.Dataset.from_tensor_slices(tf.expand_dims(self.LDCT_images, axis=-1))
        ndct_set = tf.data.Dataset.from_tensor_slices(tf.expand_dims(self.NDCT_images, axis=-1))

        return ldct_set, ndct_set
