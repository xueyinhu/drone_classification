# head-last


import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ELU


class AttentionBasic(Model):

    def __init__(self, c, h, w):
        super(AttentionBasic, self).__init__()
        self.s1 = Sequential([
            Conv2D(1, (h, w), padding="same"),
            BatchNormalization(),
            ELU()
        ])
        self.s2 = Sequential([
            Conv2D(c, (h, 1)),
            BatchNormalization(),
            ELU()
        ])
        self.s3 = Sequential([
            Conv2D(c, (1, w)),
            BatchNormalization(),
            ELU()
        ])
        self.bt = Sequential([
            BatchNormalization(),
            ELU(),
            Conv2D(c, (3, 3), padding='same', groups=c),
            BatchNormalization(),
            ELU(),
            Conv2D(c, (1, 1)),
            BatchNormalization(),
            ELU()
        ])
        self.oh = tf.ones((h, w, c))

    def call(self, x):
        oh = self.oh * self.s1(x) * self.s2(x) * self.s3(x)
        return x / 2. + x * self.bt(oh) / 2.