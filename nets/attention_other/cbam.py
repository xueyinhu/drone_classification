import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers


class ChannelAttention(Model):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg = layers.GlobalAveragePooling2D()
        self.max = layers.GlobalMaxPooling2D()
        self.conv1 = layers.Conv2D(in_planes // ratio, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(5e-4), use_bias=True, activation=tf.nn.relu)
        self.conv2 = layers.Conv2D(in_planes, kernel_size=1, strides=1, padding='same', kernel_regularizer=regularizers.l2(5e-4), use_bias=True)

    def call(self, x):
        avg = self.avg(x)
        max = self.max(x)
        avg = layers.Reshape((1, 1, avg.shape[1]))(avg)
        max = layers.Reshape((1, 1, max.shape[1]))(max)
        avg_out = self.conv2(self.conv1(avg))
        max_out = self.conv2(self.conv1(max))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)
        return out


class SpatialAttention(Model):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = layers.Conv2D(1, kernel_size=kernel_size, strides=1, activation=tf.nn.sigmoid, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(5e-4))

    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=3)
        max_out = tf.reduce_max(x, axis=3)
        out = tf.stack([avg_out, max_out], axis=3)
        out = self.conv1(out)
        return out


class CBAM_Block(Model):
    def __init__(self, c):
        super(CBAM_Block, self).__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()

    def call(self, x):
        y = self.ca(x) * x
        y = self.sa(y) * y
        return y


# m = tf.keras.Sequential([
#     tf.keras.Input(shape=(120, 40, 64)),
#     CBAM_Block(64)
# ])
# m.summary()
