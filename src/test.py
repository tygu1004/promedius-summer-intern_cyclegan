import cycle_identity_module as md
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from easydict import EasyDict
from collections import namedtuple
import inout_util as io_util

args = EasyDict({
    'dcm_path': '/data','train_patient_no_A': ['001_B30', '001_B50'], 'train_patient_no_B': ['001_B30', '001_B50'],
     'test_patient_no_A': ['L067', 'L291'], 'test_patient_no_B': ['L067', 'L291'], 'taskID': None,
      'checkpoint_dir': '/data/CYCLEIDENT/checkpoint', 'test_npy_save_dir': '/data/CYCLEIDENT/test', 'patch_size': 56,
       'whole_size': 512, 'img_channel': 1, 'img_vmax': 3072, 'img_vmin': -1024, 'model': 'cycle_identity', 'phase': 'train',
        'end_epoch': 160, 'decay_epoch': 100, 'lr': 0.0002, 'batch_size': 10, 'L1_lambda': 10.0, 'L1_gamma': 5.0, 'beta1': 0.5,
         'beta2': 0.999, 'ngf': 128, 'nglf': 15, 'ndf': 64, 'steps_per_epoch': 5000, 'save_freq': 4756, 'print_freq': 200, 'continue_train': True, 'gpu_no': 0, 'unpair': True
})
OPTIONS = namedtuple('OPTIONS', 'gf_dim glf_dim df_dim \
                              img_channel is_training')
options = OPTIONS._make((args.ngf, args.nglf, args.ndf,
                              args.img_channel, args.phase == 'train'))

with tf.device('/CPU:0'):
    #tmp = np.random.randn(32, 56, 56, 1)
    '''
    inputs1 = tf.keras.Input(shape=(56, 56, 1))
    
    inputs2 = tf.keras.Input(shape=(56, 56, 1))
    

    x = md.discriminator(inputs1, options)
    y = md.generator(inputs2, options)

    model1 = Model(inputs1, x)
    model1.summary()

    model2 = Model(inputs2, y)
    model2.summary()
    '''

    data = io_util.DCMDataLoader(args.dcm_path,\
         image_size = args.whole_size, patch_size = args.patch_size, depth = args.img_channel,
         image_max = args.img_vmax, image_min = args.img_vmin, batch_size = args.batch_size, \
         is_unpair = args.unpair, model = args.model)
    data(args.train_patient_no_A, args.train_patient_no_B)

    #print(type(data.LDCT_images[0][0: args.batch_size]))
    #print(data.LDCT_images[0][0: args.batch_size].shape)
    #print(np.expand_dims(data.LDCT_images[0][0: args.batch_size], axis=-1))
    x, y = data.get_random_patches(52)

    for i, c in enumerate(x):
        print(i)
        print(c)
    print(x)
    print(y)