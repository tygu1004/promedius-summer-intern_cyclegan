import cycle_identity_module as md
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from easydict import EasyDict
from collections import namedtuple

args = EasyDict({
    'dcm_path': '/data/CT_image','train_patient_no_A': ['L067', 'L291'], 'train_patient_no_B': ['L067', 'L291'], 'test_patient_no_A': ['L067', 'L291'], 'test_patient_no_B': ['L067', 'L291'], 'taskID': None, 'checkpoint_dir': '/data/CYCLEIDENT/checkpoint', 'test_npy_save_dir': '/data/CYCLEIDENT/test', 'patch_size': 56, 'whole_size': 512, 'img_channel': 1, 'img_vmax': 3072, 'img_vmin': -1024, 'model': 'cycle_identity', 'phase': 'train', 'end_epoch': 160, 'decay_epoch': 100, 'lr': 0.0002, 'batch_size': 10, 'L1_lambda': 10.0, 'L1_gamma': 5.0, 'beta1': 0.5, 'beta2': 0.999, 'ngf': 128, 'nglf': 15, 'ndf': 64, 'steps_per_epoch': 5000, 'save_freq': 4756, 'print_freq': 200, 'continue_train': True, 'gpu_no': 0, 'unpair': True
})
OPTIONS = namedtuple('OPTIONS', 'gf_dim glf_dim df_dim \
                              img_channel is_training')
options = OPTIONS._make((args.ngf, args.nglf, args.ndf,
                              args.img_channel, args.phase == 'train'))

with tf.device('/CPU:0'):
	tmp = np.random.randn(32, 56, 56, 1)

	inputs = tf.keras.Input(shape=(56,56,1))
	x = md.discriminator(inputs, options)
	
	model = Model(inputs, x)
	model.summary()