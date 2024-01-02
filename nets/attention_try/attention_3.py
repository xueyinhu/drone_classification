# head-last


import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ELU


class AttentionBasic(Model):

    def __init__(self, c, h, w):
        super(AttentionBasic, self).__init__()
        self.s1 = Sequential([
            Conv2D(c, (3, 3), padding='same'),
            BatchNormalization(),
            ELU(),
            Conv2D(1, (h, w), padding="same"),
            Conv2D(1, (h // 4, w // 4), padding='same'),
            BatchNormalization(),
            ELU()
        ])
        self.s2 = Sequential([
            Conv2D(c, (3, w), padding='same', groups=c // 4),
            BatchNormalization(),
            ELU(),
            Conv2D(c, (h, 1)),
            Conv2D(c, (1, w // 4), padding='same', groups=c // 4),
            BatchNormalization(),
            ELU()
        ])
        self.s3 = Sequential([
            Conv2D(c, (h, 3), padding='same', groups=c // 4),
            BatchNormalization(),
            ELU(),
            Conv2D(c, (1, w)),
            Conv2D(c, (h // 4, 1), padding='same', groups=c // 4),
            BatchNormalization(),
            ELU()
        ])

    def call(self, x):
        y = self.s1(x) * x
        y = self.s2(y) * y
        y = self.s3(y) * y
        return y

