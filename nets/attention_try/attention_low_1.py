# self.head = Sequential([
#     layers.Conv2D(8, (7, 7), (2, 2), 'same'),
#     layers.BatchNormalization(),
#     layers.ELU(),
#     MyNetHeadBlock(8),
#     layers.Conv2D(16, (5, 5), (2, 2), 'same', groups=8),

#     AttentionBasic(16, 240, 80),

#     layers.BatchNormalization(),
#     layers.ELU(),            
#     MyNetHeadBlock(16),
#     layers.Conv2D(32, (5, 5), (2, 2), 'same', groups=16),
#     layers.BatchNormalization(),
#     layers.ELU(),
#     MyNetHeadBlock(32),
#     layers.Conv2D(64, (11, 5), (3, 1), 'same', groups=32),

#     AttentionBasic(64, 40, 40),

#     layers.BatchNormalization(),
#     layers.ELU(),
# ])


import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ELU


class AttentionBasic(Model):

    def __init__(self, c, h, w):
        super(AttentionBasic, self).__init__()
        self.s1 = Sequential([
            # Conv2D(c, (3, 3), padding='same'),
            # BatchNormalization(),
            # ELU(),
            Conv2D(1, (h // 4, w // 4), padding="same"),
            Conv2D(1, (h // 8, w // 8), padding='same'),
            BatchNormalization(),
            ELU()
        ])
        self.s2 = Sequential([
            # Conv2D(c, (3, w), padding='same', groups=c // 4),
            # BatchNormalization(),
            # ELU(),
            Conv2D(c, (h // 8, 1), (h // 8, 1), padding="same", groups=c // 4),
            Conv2D(c, (8, 1)),
            Conv2D(c, (1, w // 4), padding='same', groups=c // 4),
            BatchNormalization(),
            ELU()
        ])
        self.s3 = Sequential([
            # Conv2D(c, (h, 3), padding='same', groups=c // 4),
            # BatchNormalization(),
            # ELU(),
            Conv2D(c, (1, w // 8), (1, w // 8), padding="same", groups=c // 4),
            Conv2D(c, (1, 8)),
            Conv2D(c, (h // 4, 1), padding='same', groups=c // 4),
            BatchNormalization(),
            ELU()
        ])

    def call(self, x):
        y = self.s1(x) * x
        y = self.s2(y) * y
        y = self.s3(y) * y
        return y

