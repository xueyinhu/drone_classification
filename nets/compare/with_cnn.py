import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

from nets.utils import *


class MyNetHeadBlock(Model):
    def __init__(self, c):
        super().__init__()
        self.b1 = Sequential([
            layers.Conv2D(c, (1, 1)),
            layers.Conv2D(c, (3, 3), padding='same', groups=c),
            layers.Conv2D(c, (3, 3), padding='same', groups=c),
            layers.BatchNormalization(),
            layers.ELU()
        ])
        self.b2 = Sequential([
            layers.Conv2D(c, (1, 1)),
            layers.Conv2D(c, (3, 3), padding='same', groups=c, dilation_rate=(2, 2)),
            layers.Conv2D(c, (3, 3), padding='same', groups=c),
            layers.BatchNormalization(),
            layers.ELU()
        ])
        self.b3 = Sequential([
            layers.Conv2D(c, (1, 1)),
            layers.Conv2D(c, (3, 3), padding='same', groups=c),
            layers.Conv2D(c, (3, 3), padding='same', groups=c, dilation_rate=(2, 2)),
            layers.BatchNormalization(),
            layers.ELU()
        ])
        self.b4 = Sequential([
            layers.Conv2D(c, (1, 1)),
            layers.Conv2D(c, (3, 3), padding='same', groups=c, dilation_rate=(2, 2)),
            layers.Conv2D(c, (3, 3), padding='same', groups=c, dilation_rate=(2, 2)),
            layers.BatchNormalization(),
            layers.ELU()
        ])

    def call(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return tf.concat([y1, y2, y3, y4], axis=-1)


class CompareNetCNN(Model):
    def __init__(self, config):
        super().__init__()
        self.head = Sequential([
            layers.Conv2D(8, (7, 7), (2, 2), 'same'),
            layers.BatchNormalization(),
            layers.ELU(),
            MyNetHeadBlock(8),

            layers.Conv2D(16, (1, 1)),

            layers.Conv2D(16, (5, 5), (2, 2), 'same', groups=8),
            layers.BatchNormalization(),
            layers.ELU(),
            MyNetHeadBlock(16),

            layers.Conv2D(32, (1, 1)),

            layers.Conv2D(32, (5, 5), (2, 2), 'same', groups=16),
            layers.BatchNormalization(),
            layers.ELU(),
            MyNetHeadBlock(32),

            layers.Conv2D(64, (1, 1)),
            ResM1(64, 120, 40),
            ResM1(64, 120, 40),
            ResM1(64, 120, 40),

            layers.Conv2D(64, (11, 5), (3, 1), 'same', groups=32),
            layers.BatchNormalization(),
            layers.ELU(),
        ])
        self.body = Sequential([
            layers.Conv2D(384 + 64, (3, 3), (2, 2), 'same'),
            layers.Conv2D(768 + 128, (3, 3), (2, 2), 'same'),
        ])
        self.tail = Sequential([
            layers.Conv2D((768 + 128) * 2, (5, 5), (2, 2), groups=(768 + 128)),
            layers.Conv2D((768 + 128) * 2, (3, 3), groups=(768 + 128) * 2),
            layers.Flatten(),
            layers.Dropout(.3),
            layers.Dense(5),
            layers.Softmax()
        ])

    def call(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.tail(y)
        return y

