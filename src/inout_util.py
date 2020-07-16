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
import pydicom
from datetime import datetime


# psnr
def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def tf_psnr(img1, img2, PIXEL_MAX=255.0):
    mse = tf.math.reduce_mean((img1 - img2) ** 2)
    if mse == 0:
        return tf.constant(100, dtype=tf.float32)
    return 20 * log10(tf.constant(PIXEL_MAX, dtype=tf.float32) / tf.math.sqrt(mse))


def ParseBoolean(b):
    b = b.lower()
    if b == 'true':
        return True
    elif b == 'false':
        return False
    else:
        raise ValueError('Cannot parse string into boolean.')


# argparser string -> boolean type
def ParseList(s):
    l = []
    if s.endswith('.txt'):
        f = open(s, 'r')
        lines = f.read().splitlines()
        for line in lines:
            l.append(os.path.splitext(line)[0])
    else:
        l = s.split(',')
    return l


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
    slices = [pydicom.dcmread(s) for s in path]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixel_hu(dcm_file):
    image = dcm_file.pixel_array
    intercept = dcm_file.RescaleIntercept
    slope = dcm_file.RescaleSlope
    image = image.astype(np.int16)
    image[image == -2000] = 0
    if slope != 1:
        image = slope * image.astype(np.float32)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    image = tf.expand_dims(image, axis=-1)
    return image


def dcm_read(path):
    path = path.numpy().decode('utf-8')
    dcm_file = pydicom.dcmread(path)
    img = get_pixel_hu(dcm_file)
    return img


def read_function_dcm(fn):
    out = tf.py_function(dcm_read, [fn], tf.int16)
    return out


def read_function_png(fn):
    f = tf.io.read_file(filename=fn)
    f = tf.io.decode_png(contents=f, channels=1, dtype=tf.uint8)    # shape : W * H * ch
    return f


def get_image_name(patent_no_list, length, domain_name):
    digit = 4
    slice_nm = []

    # sorted(idx), sorted(d_idx)  -> [1, 10, 2], [ 0001, 0002, 0010]
    for patent_no in patent_no_list:
        pre_fix_nm = '{}_{}'.format(patent_no, domain_name)
        for slice_number in range(length):
            s_idx = str(slice_number)
            d_idx = '0' * (digit - len(s_idx)) + s_idx
            slice_nm.append(pre_fix_nm + '_' + d_idx)

    return slice_nm


class DCMDataLoader(object):
    def __init__(self, data_path, image_size=512, patch_size=64, image_max=3071,
                 image_min=-1024, batch_size=1, extension='dcm', phase='train'):
        # dicom file dir
        self.extension = extension
        self.data_path = data_path

        # image params
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_max = image_max
        self.image_min = image_min

        # training params
        self.batch_size = batch_size

        # CT slice name
        self.phase = phase
        self.LDCT_image_name, self.NDCT_image_name = [], []

        self.LDCT_images_size = 0
        self.NDCT_images_size = 0

    # dicom file -> numpy array
    def __call__(self, patent_no_list_A, patent_no_list_B):
        def normalize(img):
            img = tf.cast(img, tf.float32)
            img = (img - self.image_min) / (self.image_max - self.image_min)
            return img

        def get_image_dataset(patent_no_list):
            path_pattern_list = [os.path.join(self.data_path, patent_no, '*.' + self.extension) for patent_no in patent_no_list]
            p_path = tf.data.Dataset.list_files(path_pattern_list)

            # dcm
            if self.extension == 'dcm':
                p = p_path.map(read_function_dcm, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # png
            else:
                p = p_path.map(read_function_png)
            # normalization
            p = p.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            return p

        def len_image_dataset(patent_no_list):
            data_size = 0
            for patent_no in patent_no_list:
                pattern = os.path.join(self.data_path, patent_no, '*.' + self.extension)
                data_size += len(glob(pattern))
            return data_size

        self.LDCT_images = get_image_dataset(patent_no_list_A)
        self.LDCT_images_size = len_image_dataset(patent_no_list_A)

        self.NDCT_images = get_image_dataset(patent_no_list_B)
        self.NDCT_images_size = len_image_dataset(patent_no_list_B)

        if self.phase != 'train':
            self.LDCT_image_name = get_image_name(patent_no_list_A, self.LDCT_images_size, 'A')
            self.NDCT_image_name = get_image_name(patent_no_list_B, self.NDCT_images_size, 'B')

    def get_train_set(self, patch_size, whole_size=512):
        whole_h = whole_w = whole_size
        h = w = patch_size

        # patch image range
        hd, hu = h // 2, int(whole_h - np.round(h / 2))
        wd, wu = w // 2, int(whole_w - np.round(w / 2))

        # patch image center(coordinate on whole image)
        h_pc, w_pc = np.random.choice(range(hd, hu + 1)), np.random.choice(range(wd, wu + 1))

        def patching(x):
            return x[h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]

        ldct_patch_set = self.LDCT_images.map(patching, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ndct_patch_set = self.NDCT_images.map(patching, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ldct_patch_set = ldct_patch_set.batch(self.batch_size)
        ndct_patch_set = ndct_patch_set.batch(self.batch_size)

        return ldct_patch_set, ndct_patch_set

    def get_test_set(self):
        return self.LDCT_images.batch(1), self.NDCT_images.batch(1)
