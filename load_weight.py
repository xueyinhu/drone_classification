import os
import logging

import numpy as np
import tensorflow as tf

from nets import *
from config import get_config


config = get_config()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.log_level
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)

weight_paths = {
    'ECM-Net + P2 A': 'J:/c/weights/attention_test/test1/c/',
    'ECM-Net + SE A': 'J:/c/c_se/',
}

net = MyNet(config)

ckp = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=net)
ckp.restore(tf.train.latest_checkpoint(weight_paths['ECM-Net + SE A']))


