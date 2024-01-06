import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, ELU, MaxPool2D, AvgPool2D, Conv2DTranspose


class AttentionBasic_Bigger(Model):

    def __init__(self, c, h, w):
        super(AttentionBasic, self).__init__()
        self.s1 = Sequential([
            # Conv2D(c // 4, (5, 1), padding='same', groups=c // 4),
            # Conv2D(c // 4, (1, 5), padding='same', groups=c // 8),
            # BatchNormalization(),
            # ELU(),
            Conv2D(4, (h // 8, 1), padding="same"),
            Conv2D(1, (1, w // 8), padding="same"),
            BatchNormalization(),
            ELU()
        ])
        self.s2 = Sequential([
            # Conv2D(c // 4, (1, w // 4), padding='same', groups=c // 4),
            # Conv2D(c // 4, (3, 1), padding="same", groups=c // 8),
            # BatchNormalization(),
            # ELU(),
            MaxPool2D((h // 8, 1), strides=(4, 1), padding='same'),
            Conv2D(c // 4, (h // 4, 1)),
            Conv2D(c, (1, w // 4), padding='same', groups=c // 4),
            BatchNormalization(),
            ELU()
        ])
        self.s3 = Sequential([
            # Conv2D(c // 4, (h // 4, 1), padding='same', groups=c // 4),
            # Conv2D(c // 4, (1, 3), padding="same", groups=c // 8),
            # BatchNormalization(),
            # ELU(),
            MaxPool2D((1, w // 8), strides=(1, 4), padding='same'),
            Conv2D(c // 4, (1, w // 4)),
            Conv2D(c, (h // 4, 1), padding='same', groups=c // 4),
            BatchNormalization(),
            ELU()
        ])
        self.oh = tf.ones((h, w, c))

    def call(self, x):
        t1 = self.s1(x)
        t2 = self.s2(x)
        t3 = self.s3(x)
        o_ = self.oh * t1 * t2 * t3
        return x * o_ * .7 + x * .3


class AttentionBasic_Fucker(Model):

    def __init__(self, c, h, w):
        super(AttentionBasic, self).__init__()
        self.s1 = Sequential([
            Conv2D(max(c // 4, 8), (1, 1)),
            Conv2D(1, (7, 7), padding="same"),
            BatchNormalization(),
            ELU()
        ])
        self.s2_m = MaxPool2D((h, 1), (1, 1))
        self.s2_a = AvgPool2D((h, 1), (1, 1))
        self.s2 = Sequential([
            Conv2D(c, (1, 5), padding="same", groups=c),
            BatchNormalization(),
            ELU()
        ])
        self.s3_m = MaxPool2D((1, w), (1, 1))
        self.s3_a = AvgPool2D((1, w), (1, 1))
        self.s3 = Sequential([
            Conv2D(c, (5, 1), padding="same", groups=c),
            BatchNormalization(),
            ELU()
        ])
        # self.oh = tf.ones((h, w, c))

    def call(self, x):
        t1 = self.s1(x) * x
        t2 = self.s2(self.s2_m(t1) - self.s2_a(t1)) * t1
        t3 = self.s3(self.s3_m(t2) - self.s3_a(t2)) * t2
        # o_ = self.oh * t1 * t2 * t3
        return t3  * .7 + x * .3
    

class AttentionBasic_Dead(Model):

    def __init__(self, c, h, w):
        super(AttentionBasic, self).__init__()
        self.s1_m = MaxPool2D((h, w), (1, 1))
        self.s1_a = AvgPool2D((h, w), (1, 1))
        # self.s2_m = MaxPool2D((h, 1), (1, 1))
        # self.s2_a = AvgPool2D((h, 1), (1, 1))
        # self.s2_c = Conv2D(1, (1, 1))
        # self.s3_m = MaxPool2D((1, w), (1, 1))
        # self.s3_a = AvgPool2D((1, w), (1, 1))
        # self.s3_c = Conv2D(1, (1, 1))
        # self.oh = tf.ones((h, w, c))
        self.s1 = Sequential([
            Conv2D(max(8, c // 4), (1, 1)),
            BatchNormalization(),
            ELU(),
            Conv2D(c, (1, 1)),
            BatchNormalization(),
            ELU()
        ])
        # self.s2 = Sequential([
        #     Conv2D(1, (1, w), (1, 2), padding="same"),
        #     BatchNormalization(),
        #     ELU(),
        #     Conv2DTranspose(1, (1, w), (1, 2), padding="same"),
        #     BatchNormalization(),
        #     ELU()
        # ])
        # self.s3 = Sequential([
        #     Conv2D(1, (h, 1), (2, 1), padding="same"),
        #     BatchNormalization(),
        #     ELU(),
        #     Conv2DTranspose(1, (h, 1), (2, 1), padding="same"),
        #     BatchNormalization(),
        #     ELU()
        # ])

    def call(self, x):
        y = self.s1_m(x) - self.s1_a(x)
        t = self.s1(y) * x
        # y = self.s2_c(self.s2_m(t) - self.s2_a(t))
        # t = self.s2(y) * t
        # y = self.s3_c(self.s3_m(t) - self.s3_a(t))
        # t = self.s3(y) * t
        # o_ = self.oh * t1 * t2 * t3
        return t * .7 + x * .3


class AttentionBasic_(Model):

    def __init__(self, c, h, w):
        super(AttentionBasic_, self).__init__()
        self.s1_m = MaxPool2D((h, w), (1, 1))
        self.s1_a = AvgPool2D((h, w), (1, 1))
        self.s2_m = MaxPool2D((h, 1), (1, 1))
        self.s2_a = AvgPool2D((h, 1), (1, 1))
        # self.s2_c = Conv2D(1, (1, 1))
        self.s3_m = MaxPool2D((1, w), (1, 1))
        self.s3_a = AvgPool2D((1, w), (1, 1))
        # self.s3_c = Conv2D(1, (1, 1))
        # self.oh = tf.ones((h, w, c))
        self.s1 = Sequential([
            Conv2D(max(16, c // 2), (1, 1)),
            BatchNormalization(),
            ELU(),
            Conv2D(c, (1, 1)),
            # BatchNormalization(),
            # ELU()
        ])
        self.s2 = Sequential([
            Conv2D(1, (1, w // 8), (1, 4), padding="same"),
            BatchNormalization(),
            ELU(),
            Conv2DTranspose(1, (1, w // 8), (1, 2), padding="same"),
            # BatchNormalization(),
            # ELU()
        ])
        self.s3 = Sequential([
            Conv2D(1, (h // 8, 1), (4, 1), padding="same"),
            BatchNormalization(),
            ELU(),
            Conv2DTranspose(1, (h // 8, 1), (2, 1), padding="same"),
            # BatchNormalization(),
            # ELU()
        ])
        # self.se = Sequential([
        #     # BatchNormalization(),
        #     # Conv2D(c, (3, 3), padding='same', groups=c),
        #     # BatchNormalization(),
        #     # ELU(),
        #     # Conv2D(c, (3, 1), padding='same', groups=c // 2),
        #     # BatchNormalization(),
        #     # ELU(),
        #     # Conv2D(c, (1, 3), padding='same', groups=c // 2),
        #     BatchNormalization(),
        #     # ELU(),
        # ])

    def call(self, x):
        y = tf.concat([self.s1_m(x), self.s1_a(x)], axis=3)
        t = tf.sigmoid(self.s1(y)) * x
        y = tf.concat([self.s2_m(t), self.s2_a(t)], axis=2)
        t = tf.sigmoid(self.s2(y)) * t
        y = tf.concat([self.s3_m(t), self.s3_a(t)], axis=1)
        t = tf.sigmoid(self.s3(y)) * t
        # o_ = self.oh * t1 * t2 * t3
        # o_ = tf.sigmoid(o_)
        # y_ = self.se(o_)
        return t # y_ * x * .7 + x * .3
    

class AttentionBasic(Model):

    def __init__(self, c, h, w):
        super(AttentionBasic, self).__init__()
        self.s1_a = AvgPool2D((h, w), (1, 1))
        self.s2_a = AvgPool2D((h, 1), (1, 1))
        self.s3_a = AvgPool2D((1, w), (1, 1))
        self.s1 = Sequential([
            Conv2D(c, (1, 1)),
        ])
        self.s2 = Sequential([
            Conv2D(1, (1, 1)),
            Conv2D(1, (1, w // 8), padding="same"),
        ])
        self.s3 = Sequential([
            Conv2D(1, (1, 1)),
            Conv2D(1, (h // 8, 1), padding="same"),
        ])

    def call(self, x):
        y = self.s1_a(x)
        t = tf.sigmoid(self.s1(y)) * x
        y = self.s2_a(t)
        t = tf.sigmoid(self.s2(y)) * t
        y = self.s3_a(t)
        t = tf.sigmoid(self.s3(y)) * t
        return t


# m = Sequential([
#     tf.keras.Input(shape=(40, 40, 64)),
#     AttentionBasic(64, 40, 40)
# ])
# m.summary()

# m = Sequential([
#     tf.keras.Input(shape=(120, 40, 64)),
#     AttentionBasic(64, 120, 40)
# ])
# m.summary()

# m = Sequential([
#     tf.keras.Input(shape=(240, 80, 32)),
#     AttentionBasic(32, 240, 80)
# ])
# m.summary()

# m = Sequential([
#     tf.keras.Input(shape=(480, 160, 16)),
#     AttentionBasic(16, 480, 160)
# ])
# m.summary()
