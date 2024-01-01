
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, AvgPool2D
 

def coord_act(x):
    tmpx = tf.nn.relu6(x+3) / 6
    x = x * tmpx
    return x


class Coord_Block(Model):
    def __init__(self, c, h, w, reduction = 32):
        super().__init__()
        self.xh = AvgPool2D(pool_size=(1, w), strides = 1)
        self.xw = AvgPool2D(pool_size=(h, 1), strides = 1)
        mip = max(8, c // reduction)
        self.c1 = Conv2D(mip, (1, 1), strides=1, activation=coord_act,name='ca_conv1')
        self.c2 = Conv2D(c, (1, 1), strides=1,activation=tf.nn.sigmoid,name='ca_conv2')
        self.c3 = Conv2D(c, (1, 1), strides=1,activation=tf.nn.sigmoid,name='ca_conv3')

    def call(self, x):
        [b, h, w, c] = x.shape
        x_h = self.xh(x)
        x_w = self.xw(x)
        x_w = tf.transpose(x_w, [0, 2, 1, 3])
        y = tf.concat([x_h, x_w], axis=1)
        y = self.c1(y)
        x_h, x_w = tf.split(y, num_or_size_splits=2, axis=1)
        x_w = tf.transpose(x_w, [0, 2, 1, 3])
        a_h = self.c2(x_h)
        a_w = self.c3(x_w)
        out = x * a_h * a_w
        return out
