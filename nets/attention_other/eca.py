import tensorflow as tf
from tensorflow.keras import layers


class ECA_Block(layers.Layer):
    def __init__(self):
        super(ECA_Block, self).__init__()
        self.gap = layers.GlobalAvgPool2D()
        self.conv = layers.Conv1D(1, kernel_size=3, padding='same', use_bias=False)
        self.sigmoid = tf.sigmoid
        self.reshape_1 = layers.Reshape((-1, 1))
        self.reshape_2 = layers.Reshape((1, 1, -1))

    def call(self, x):
        y = self.gap(x) 
        y = self.reshape_1(y)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = self.reshape_2(y)
        return x * y
    

# m = tf.keras.Sequential([
#     tf.keras.Input(shape=(120, 40, 64)),
#     ECA_Block()
# ])
# m.summary()
