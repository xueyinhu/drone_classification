import tensorflow as tf

from nets import MyNet, ShuffleNetV2
from config import get_config


m = tf.keras.Sequential([
    tf.keras.Input(shape=(960, 320, 3)),
    MyNet(get_config())
])
m.summary()