import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, sz , w_offset=0, h_offset=0, is_random=True):
    #print(x.shape)
    x= crop(x, wrg=sz, hrg=sz, w_offset=0, h_offset=0, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[384, 384], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def normalize_fn(x):
    #print(x.shape)
    #x= crop(x, wrg=sz, hrg=sz, w_offset=0, h_offset=0, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x
