import tensorflow as tf

from nets import MyNet, ShuffleNetV2, MobileNetV2, InceptionV3
from config import get_config


m = tf.keras.Sequential([
    tf.keras.Input(shape=(960, 320, 3)),
    MyNet(get_config())
])
m.summary()