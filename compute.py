import tensorflow as tf

from nets import *
from config import get_config


i = tf.keras.Input((960, 320, 3))
m = tf.keras.Model(i, MyNet(get_config())(i))
# m = InceptionV3(get_config())
# m.build((960, 320, 3))
m.summary()
