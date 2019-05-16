from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.patch_size = 50
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-3
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 100
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.ori_img_path = 'data/train_ori/'
config.TRAIN.haze_img_path = 'data/train_haze/'

config.VALID = edict()
## test set location
config.VALID.ori_img_path = 'data/valid_ori/'
config.VALID.haze_img_path = 'data/valid_haze/'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
