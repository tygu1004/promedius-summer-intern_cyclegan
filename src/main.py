# -*- coding: utf-8 -*-
"""
Module:    main.py
Language:  Python3
Date:      2020-06-30 14:16:00
Version:   open.VER 1.0
Developer: LEE GAEUN  (ggelee93@gmail.com) / @author: yeohyeongyu

Copyright (c) Promedius.
All rights reserved.
"""

import argparse
import os
from cycle_identity_model import cycle_identity
import inout_util as ut

parser = argparse.ArgumentParser(description='')
# -------------------------------------
# set load directory
parser.add_argument('--data_path', dest='data_path', help='data file directory', required=True)
parser.add_argument('--extension', dest='extension', default='dcm', help='file extension', required=True)
parser.add_argument('--train_A_list', dest='train_patient_no_A', type=ut.ParseList, required=True)
parser.add_argument('--train_B_list', dest='train_patient_no_B', type=ut.ParseList, required=True)
parser.add_argument('--test_A_list', dest='test_patient_no_A', type=ut.ParseList, required=True)
parser.add_argument('--test_B_list', dest='test_patient_no_B', type=ut.ParseList, required=True)
parser.add_argument('--taskID', dest='taskID', default=None,
                    help='A unique ID for the log. It is a required input in the test phase.')

# set save directory
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='/data/CYCLEIDENT/checkpoint',
                    help='check point dir')
parser.add_argument('--test_npy_save_dir', dest='test_npy_save_dir', default='/data/CYCLEIDENT/test',
                    help='test numpy file save dir')

# image info
parser.add_argument('--patch_size', dest='patch_size', type=int, default=56, help='image patch size, h=w')
parser.add_argument('--whole_size', dest='whole_size', type=int, default=512, help='image whole size, h=w')
parser.add_argument('--img_channel', dest='img_channel', type=int, default=1, help='image channel, 1')
parser.add_argument('--img_vmax', dest='img_vmax', type=int, default=3072, help='max value in image')
parser.add_argument('--img_vmin', dest='img_vmin', type=int, default=-1024, help='max value in image')

# train, test
parser.add_argument('--phase', dest='phase', default='train', help='train, test')

# train detail
parser.add_argument('--epoch', dest='epoch', type=int, default=160, help='set epoch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='batch size')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight of cyclic loss')
parser.add_argument('--L1_gamma', dest='L1_gamma', type=float, default=5.0, help='weight of identity loss')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5,
                    help='The exponential decay rate for the 1st moment estimates.')
parser.add_argument('--beta2', dest='beta2', type=float, default=0.999,
                    help='The exponential decay rate for the 2nd moment estimates.')
parser.add_argument('--ngf', dest='ngf', type=int, default=128, help='# of gen filters in first conv layer')
parser.add_argument('--nglf', dest='nglf', type=int, default=15, help='# of gen filters in last conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')

# others
parser.add_argument('--save_freq', dest='save_freq', type=int, default=2378 * 2,
                    help='save a model every save_freq (iteration)')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100 * 2, help='print_freq (iterations)')
parser.add_argument('--continue_train', dest='continue_train', type=ut.ParseBoolean, default=True,
                    help='load the latest model: true, false')
parser.add_argument('--gpu_no', dest='gpu_no', type=int, default=0, help='gpu no')

# -------------------------------------
args = parser.parse_args()
print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)

model = cycle_identity(args)
model.train(args) if args.phase == 'train' else model.test(args)
