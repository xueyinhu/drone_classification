# head-last before-BN


import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ELU, MaxPool2D


class AttentionBasic(Model):

    def __init__(self, c, h, w):
        super(AttentionBasic, self).__init__()
        self.s1 = Sequential([
            Conv2D(c // 4, (5, 1), padding='same', groups=c // 4),
            Conv2D(c // 4, (1, 5), padding='same', groups=c // 8),
            BatchNormalization(),
            ELU(),
            Conv2D(4, (h // 8, 1), padding="same"),
            Conv2D(1, (1, w // 8), padding="same"),
            BatchNormalization(),
            ELU()
        ])
        self.s2 = Sequential([
            Conv2D(c // 4, (1, w // 4), padding='same', groups=c // 4),
            Conv2D(c // 4, (3, 1), padding="same", groups=c // 8),
            BatchNormalization(),
            ELU(),
            MaxPool2D((h // 8, 1), strides=(4, 1), padding='same'),
            Conv2D(c // 4, (10, 1)),
            Conv2D(c, (1, w // 4), padding='same', groups=c // 4),
            BatchNormalization(),
            ELU()
        ])
        self.s3 = Sequential([
            Conv2D(c // 4, (h // 4, 1), padding='same', groups=c // 4),
            Conv2D(c // 4, (1, 3), padding="same", groups=c // 8),
            BatchNormalization(),
            ELU(),
            MaxPool2D((1, w // 8), strides=(1, 4), padding='same'),
            Conv2D(c // 4, (1, 10)),
            Conv2D(c, (h // 4, 1), padding='same', groups=c // 4),
            BatchNormalization(),
            ELU()
        ])

    def call(self, x):
        y = self.s1(x) * x
        y = self.s2(y) * y
        y = self.s3(y) * y
        return y