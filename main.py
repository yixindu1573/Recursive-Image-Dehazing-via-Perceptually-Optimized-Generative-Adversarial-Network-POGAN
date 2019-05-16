#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config
from os import listdir


###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
patch_sz = config.patch_size
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))


def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint_init"  # checkpoint_resize_conv
    #tl.files.exists_or_mkdir(checkpoint_dir)

    checkpoint_d_dir = "checkpoint_d"  # checkpoint_resize_conv
    #tl.files.exists_or_mkdir(checkpoint_d_dir)

    checkpoint_g_dir = "checkpoint_g"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_g_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_ori_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.ori_img_path, regx='.*.png', printable=False))
    train_haze_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.haze_img_path, regx='.*.png', printable=False))
    train_ori_imgs = tl.vis.read_images(train_ori_img_list, path=config.TRAIN.ori_img_path, n_threads=32)
    train_haze_imgs = tl.vis.read_images(train_haze_img_list, path=config.TRAIN.haze_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, patch_sz, patch_sz, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, patch_sz, patch_sz, 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    net_d.print_params(False)


    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)


    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')

    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)


    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)


    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)

    idx_num = len(train_ori_imgs)%batch_size;
    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init + 1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0
        for idx in range(0, len(train_ori_imgs)-idx_num, batch_size):
            b_imgs_ori = []
            b_imgs_haze = []
            step_time = time.time()
            for idx_sub in range(0,batch_size):
                h, w, c = train_ori_imgs[idx_sub+idx].shape
                h_offset = int(np.random.uniform(0, h - patch_sz) - 1)
                w_offset = int(np.random.uniform(0, w - patch_sz) - 1)
                ori_crop = tl.prepro.threading_data(train_ori_imgs[idx_sub+idx:idx_sub+idx+1], fn=crop_sub_imgs_fn, sz=patch_sz, w_offset = w_offset, h_offset = h_offset, is_random=False)
                haze_crop = tl.prepro.threading_data(train_haze_imgs[idx_sub+idx:idx_sub+idx+1], fn=crop_sub_imgs_fn, sz=patch_sz, w_offset = w_offset, h_offset = h_offset, is_random=False)
                b_imgs_ori.append(ori_crop[0])
                b_imgs_haze.append(haze_crop[0])
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_haze, t_target_image: b_imgs_ori})
            # print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)
        ## save model
    tl.files.save_npz(net_g.all_params, name=checkpoint_g_dir + '/g_'+ '%03d'%(epoch)+'.npz', sess=sess)


    ###========================= train GAN (SRGAN) =========================###
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        for idx in range(0, len(train_ori_imgs)-idx_num, batch_size):
            step_time = time.time()

            b_imgs_ori = []
            b_imgs_haze = []
            for idx_sub in range(0,batch_size):
                h, w, c = train_ori_imgs[idx_sub+idx].shape
                h_offset = int(np.random.uniform(0, h - patch_sz) - 1)
                w_offset = int(np.random.uniform(0, w - patch_sz) - 1)
                ori_crop = tl.prepro.threading_data(train_ori_imgs[idx_sub+idx:idx_sub+idx+1], fn=crop_sub_imgs_fn, sz=patch_sz, w_offset = w_offset, h_offset = h_offset, is_random=False)
                haze_crop = tl.prepro.threading_data(train_haze_imgs[idx_sub+idx:idx_sub+idx+1], fn=crop_sub_imgs_fn, sz=patch_sz, w_offset = w_offset, h_offset = h_offset, is_random=False)
                b_imgs_ori.append(ori_crop[0])
                b_imgs_haze.append(haze_crop[0])

            ## update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_haze, t_target_image: b_imgs_ori})
            ## update G
            errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: b_imgs_haze, t_target_image: b_imgs_ori})
            #print("Epoch [%2d/%2d] %4d time: %4.4fmins, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %(epoch, n_epoch, n_iter, (time.time() - step_time)/60, errD, errG, errM, errV, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)

        tl.files.save_npz(net_g.all_params, name=checkpoint_g_dir + '/g_'+ '%03d'%(epoch)+'.npz', sess=sess)
        #tl.files.save_npz(net_d.all_params, name=checkpoint_d_dir + '/d_'+ '%03d'%(epoch)+'.npz', sess=sess)



def e():
    ## create folders to save result images
    save_dir = "result"
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint_g"
    ###====================== PRE-LOAD DATA ===========================###
    valid_haze_img_list = sorted(tl.files.load_file_list(path=config.VALID.haze_img_path, regx='.*.png', printable=False))
    valid_haze_imgs = tl.vis.read_images(valid_haze_img_list, path=config.VALID.haze_img_path, n_threads=32)
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + 'model.npz', network=net_g)
    ###========================== DEFINE MODEL ============================###
    for imid in range(0,len(valid_haze_imgs)):
        print (valid_haze_img_list[imid])
        valid_haze_img = valid_haze_imgs[imid]
        valid_haze_img = (valid_haze_img / 127.5) - 1  # rescale to ［－1, 1]
        size = valid_haze_img.shape
        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_haze_img]})
        tl.vis.save_image(out[0], save_dir + '/' + valid_haze_img_list[imid])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'e':
        e()
    elif tl.global_flag['mode'] == 'ec':
        ec()
    else:
        raise Exception("Unknow --mode")
