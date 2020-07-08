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
    'dcm_path': '/data', 'train_patient_no_A': ['001_B30'], 'train_patient_no_B': ['001_B30'],
    'test_patient_no_A': ['001_B30'], 'test_patient_no_B': ['001_B50'], 'taskID': None,
    'checkpoint_dir': '/data/CYCLEIDENT/checkpoint', 'test_npy_save_dir': '/data/CYCLEIDENT/test', 'patch_size': 30,
    'whole_size': 512, 'img_channel': 1, 'img_vmax': 3072, 'img_vmin': -1024, 'model': 'cycle_identity',
    'phase': 'train',
    'end_epoch': 160, 'decay_epoch': 100, 'lr': 0.0002, 'batch_size': 2, 'L1_lambda': 10.0, 'L1_gamma': 5.0,
    'beta1': 0.5,
    'beta2': 0.999, 'ngf': 128, 'nglf': 15, 'ndf': 64, 'steps_per_epoch': 5000, 'save_freq': 4756, 'print_freq': 200,
    'continue_train': True, 'gpu_no': 0, 'unpair': True
})
OPTIONS = namedtuple('OPTIONS', 'gf_dim glf_dim df_dim \
                              img_channel is_training')
options = OPTIONS._make((args.ngf, args.nglf, args.ndf,
                         args.img_channel, args.phase == 'train'))

with tf.device('/CPU:0'):
    '''
    tmp = np.random.randn(56, 56, 1)
    tmp2 = np.random.randn(2, 56, 56, 1)

    x = md.discriminator(tmp.shape, options)
    y = md.generator(tmp.shape, options)

    print(x(tmp2).shape)
    print(y(tmp2).shape)
    '''

    '''
    data = io_util.DCMDataLoader(args.dcm_path,\
         image_size = args.whole_size, patch_size = args.patch_size, depth = args.img_channel,
         image_max = args.img_vmax, image_min = args.img_vmin, batch_size = args.batch_size, \
         is_unpair = args.unpair, model = args.model)
    data(args.train_patient_no_A, args.train_patient_no_B)

    #print(type(data.LDCT_images[0][0: args.batch_size]))
    #print(data.LDCT_images[0][0: args.batch_size].shape)
    #print(np.expand_dims(data.LDCT_images[0][0: args.batch_size], axis=-1))
    x, y = data.get_random_patches(256)

    '''
    '''
    for i, c in enumerate(x):
        print(i)
        c = np.expand_dims(c, axis=-1)
        print(c.shape)
        tf.keras.preprocessing.image.save_img("data/save"+str(i)+".png", c)
    '''
    model = cycle_identity(args)
    model.train(args)
