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
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
import dicom
from random import shuffle
import threading
from datetime import datetime


class DCMDataLoader(object):
    def __init__(self, dcm_path, \
                 image_size=512, patch_size=64, depth=1, \
                 image_max=3071, image_min=-1024, batch_size=1, \
                 is_unpair=False, model='', num_threads=1, extension='dcm'):

        # dicom file dir
        self.extension = extension
        self.dcm_path = dcm_path

        # image params
        self.image_size = image_size
        self.patch_size = patch_size
        self.depth = depth

        self.image_max = image_max
        self.image_min = image_min

        # training params
        self.batch_size = batch_size
        self.is_unpair = is_unpair
        self.model = model

        # CT slice name
        self.LDCT_image_name, self.NDCT_image_name = [], []

        # batch generator  prameters
        self.num_threads = num_threads
        self.capacity = 20 * self.num_threads * self.batch_size
        self.min_queue = 10 * self.num_threads * self.batch_size

    # dicom file -> numpy array
    def __call__(self, patent_no_list_A, patent_no_list_B):

        def get_images(patent_no_list, domain_name):
            p = []
            image_name = []
            for patent_no in (patent_no_list):
                p_path = glob(os.path.join(self.dcm_path, patent_no, '*.' + self.extension))

                # load images
                org_images, slice_nm = self.get_pixels_hu(self.load_scan(p_path),
                                                          '{}_{}'.format(patent_no, domain_name))

                # CT slice name
                image_name.extend(slice_nm)

                # normalization
                p.append(self.normalize(org_images, self.image_max, self.image_min))

            return p, image_name

        p_LDCT, self.LDCT_image_name = get_images(patent_no_list_A, 'A')
        p_NDCT, self.NDCT_image_name = get_images(patent_no_list_B, 'B')

        self.LDCT_images = np.concatenate(tuple(p_LDCT), axis=0)
        self.NDCT_images = np.concatenate(tuple(p_NDCT), axis=0)

        # image index
        self.LDCT_index, self.NDCT_index = list(range(len(self.LDCT_images))), list(range(len(self.NDCT_images)))

    def load_scan(self, path):
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

    def get_pixels_hu(self, slices, pre_fix_nm=''):
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

    def normalize(self, img, max_=3071, min_=-1024):
        img = img.astype(np.float32)
        if 'cycle' in self.model:  # -1 ~ 1
            img = 2 * ((img - min_) / (max_ - min_)) - 1
            return img
        else:  # 0 ~ 1
            img = (img - min_) / (max_ - min_)
            return img

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

    def input_pipeline(self, sess, image_size, end_point, depth=1):
        queue_input = tf.placeholder(tf.float32)
        queue_output = tf.placeholder(tf.float32)
        queue = tf.FIFOQueue(capacity=self.capacity, dtypes=[tf.float32, tf.float32], \
                             shapes=[(image_size, image_size, depth), (image_size, image_size, depth)])
        enqueue_op = queue.enqueue_many([queue_input, queue_output])
        close_op = queue.close()
        dequeue_op = queue.dequeue_many(self.batch_size)

        self.coord = tf.train.Coordinator()

        def enqueue(coord):
            enqueue_size = max(200, self.batch_size)

            if self.model == 'cyclegan':  # only cyclegan (cycelgain-identity:random patch))
                '''
                self.step = 0
                while not coord.should_stop():
                    start_pos = 0
                    if self.is_unpair:
                        shuffle(self.LDCT_index)
                        shuffle(self.NDCT_index)
                    else:
                        self.NDCT_index = self.LDCT_index
                        shuffle(self.LDCT_index)

                    while start_pos < len(self.LDCT_index):
                        end_pos = start_pos + enqueue_size
                        raw_LDCT_chunk = self.LDCT_images[self.LDCT_index][start_pos: end_pos]
                        raw_NDCT_chunk = self.NDCT_images[self.NDCT_index][start_pos: end_pos]

                        sess.run(enqueue_op, feed_dict={queue_input: np.expand_dims(raw_LDCT_chunk, axis=-1), \
                                                        queue_output: np.expand_dims(raw_NDCT_chunk, axis=-1)})
                        start_pos += enqueue_size
                    self.step += 1
                if self.step > end_point:
                    coord.request_stop()
                sess.run(close_op)
                '''
            else:
                self.step = 0
                while not coord.should_stop():
                    LDCT_imgs, NDCT_imgs = [], []
                    for i in range(enqueue_size):
                        if self.is_unpair:
                            L_sltd_idx = np.random.choice(self.LDCT_index)
                            N_sltd_idx = np.random.choice(self.NDCT_index)
                        else:
                            L_sltd_idx = N_sltd_idx = np.random.choice(self.LDCT_index)

                        pat_LDCT, pat_NDCT = \
                            self.get_randam_patches(self.LDCT_images[L_sltd_idx],
                                                    self.NDCT_images[N_sltd_idx], image_size)
                        LDCT_imgs.append(np.expand_dims(pat_LDCT, axis=-1))
                        NDCT_imgs.append(np.expand_dims(pat_NDCT, axis=-1))
                    sess.run(enqueue_op, feed_dict={queue_input: np.array(LDCT_imgs), \
                                                    queue_output: np.array(NDCT_imgs)})
                    self.step += 1
                if self.step > end_point:
                    coord.request_stop()
                sess.run(close_op)

        self.enqueue_threads = [threading.Thread(target=enqueue, args=(self.coord,)) for i in range(self.num_threads)]
        for t in self.enqueue_threads: t.start()

        return dequeue_op


# ROI crop
def ROI_img(whole_image, row=[200, 350], col=[75, 225]):
    patch_ = whole_image[row[0]:row[1], col[0]: col[1]]
    return np.array(patch_)


# psnr
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def tf_psnr(img1, img2, PIXEL_MAX=255.0):
    mse = tf.reduce_mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * log10(PIXEL_MAX / tf.sqrt(mse))


def psnr(img1, img2, PIXEL_MAX=255.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# save mk img
def save_image(LDCT, NDCT, output_, save_dir='.', max_=1, min_=0):
    f, axes = plt.subplots(2, 3, figsize=(30, 20))

    axes[0, 0].imshow(LDCT, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[0, 1].imshow(NDCT, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[0, 2].imshow(output_, cmap=plt.cm.gray, vmax=max_, vmin=min_)

    axes[1, 0].imshow(NDCT.astype(np.float32) - LDCT.astype(np.float32), cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[1, 1].imshow(NDCT - output_, cmap=plt.cm.gray, vmax=max_, vmin=min_)
    axes[1, 2].imshow(output_ - LDCT, cmap=plt.cm.gray, vmax=max_, vmin=min_)

    axes[0, 0].title.set_text('LDCT image')
    axes[0, 1].title.set_text('NDCT image')
    axes[0, 2].title.set_text('output image')

    axes[1, 0].title.set_text('NDCT - LDCT  image')
    axes[1, 1].title.set_text('NDCT - outupt image')
    axes[1, 2].title.set_text('output - LDCT  image')
    if save_dir != '.':
        f.savefig(save_dir)
        plt.close()

    # ---------------------------------------------------


# argparser string -> boolean type
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
