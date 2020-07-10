import cycle_identity_module as md
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from easydict import EasyDict
from cycle_identity_model import cycle_identity
from collections import namedtuple
import inout_util as io_util

args = EasyDict({
    'dcm_path': 'data/', 'train_patient_no_A': ['001_B30'], 'train_patient_no_B': ['001_B50'],
    'test_patient_no_A': ['002_B30'], 'test_patient_no_B': ['002_B50'], 'taskID': None,
    'checkpoint_dir': 'logs/checkpoint', 'test_npy_save_dir': 'logs/results', 'patch_size': 64,
    'whole_size': 512, 'img_channel': 1, 'img_vmax': 3072, 'img_vmin': -1024,
    'phase': 'train',
    'epoch': 160, 'lr': 0.0002, 'batch_size': 8, 'L1_lambda': 10.0, 'L1_gamma': 5.0,
    'beta1': 0.5, 'beta2': 0.999, 'ngf': 128, 'nglf': 15, 'ndf': 64, 'save_freq': 30, 'print_freq': 30,
    'continue_train': True, 'gpu_no': 0
})
OPTIONS = namedtuple('OPTIONS', 'gf_dim glf_dim df_dim \
                              img_channel is_training')
options = OPTIONS._make((args.ngf, args.nglf, args.ndf,
                         args.img_channel, args.phase == 'train'))

model = cycle_identity(args)
model.train(args)