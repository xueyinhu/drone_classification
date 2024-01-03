import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

from nets.attention import AttentionBasic
from nets.attention_other import *


class ResInceptionM1(Model):
    def __init__(self, c):
        super(ResInceptionM1, self).__init__()
        self.s1 = Sequential([
            layers.Conv2D(c, (3, 3), padding='same', groups=c),
            layers.Conv2D(c, (1, 1)),
            layers.BatchNormalization(),
            layers.ELU()
        ])
        self.b1 = Sequential([
            layers.Conv2D(c // 2, (3, 1), padding='same', groups=c // 2),
            layers.Conv2D(c // 2, (3, 1), padding='same', groups=c // 2),
            layers.Conv2D(c // 2, (3, 1), padding='same', groups=c // 2),
            layers.BatchNormalization(),
            layers.ELU()
        ])
        self.b2 = Sequential([
            layers.Conv2D(c // 2, (1, 3), padding='same', groups=c // 2),
            layers.Conv2D(c // 2, (1, 3), padding='same', groups=c // 2),
            layers.Conv2D(c // 2, (1, 3), padding='same', groups=c // 2),
            layers.BatchNormalization(),
            layers.ELU()
        ])
        self.s2 = Sequential([
            layers.Conv2D(c, (3, 3), padding='same', groups=c),
            layers.Conv2D(c, (1, 1)),
            layers.BatchNormalization(),
            layers.ELU()
        ])

    def call(self, x):
        y = self.s1(x)
        p = self.b1(y)
        q = self.b2(y)
        y = self.s2(tf.concat([p, q], axis=-1))
        return y * .3 + x * .7


class ResM1(Model):
    def __init__(self, c, h, w):
        super(ResM1, self).__init__()
        self.c0 = layers.Conv2D(c, (1, 1))
        self.s1 = Sequential([
            layers.Conv2D(c, (3, 3), padding='same', groups=c),
            layers.BatchNormalization(),
            layers.ELU()
        ])
        self.s2 = Sequential([
            layers.Conv2D(c, (3, 3), padding='same', groups=c),
            layers.BatchNormalization(),
            layers.ELU()
        ])

        self.at = AttentionBasic(c, h, w)
        # self.at = SE_Block(c)
        # self.at = CBAM_Block(c)
        # self.at = Coord_Block(c, h, w)

    def call(self, x):
        y = self.c0(self.s2(self.s1(x)))
        y = self.at(y)
        return y * .7 + x * .3


# M = ResM1

# m = tf.keras.Sequential([
#     tf.keras.Input(shape=(480, 160, 16)),
#     M(16, 480, 160)
# ])
# m.summary()

# m = tf.keras.Sequential([
#     tf.keras.Input(shape=(240, 80, 32)),
#     M(32, 240, 80)
# ])
# m.summary()

# m = tf.keras.Sequential([
#     tf.keras.Input(shape=(120, 40, 64)),
#     M(64, 120, 40)
# ])
# m.summary()

# m = tf.keras.Sequential([
#     tf.keras.Input(shape=(40, 40, 64)),
#     M(64)
# ])
# m.summary()
