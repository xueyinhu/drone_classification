import tensorflow as tf

class SE_Block(tf.keras.Model):
    def __init__(self, c):
        super().__init__()
        self.a = tf.keras.layers.GlobalAveragePooling2D()
        self.d1 = tf.keras.layers.Dense(c // 2)
        self.d2 = tf.keras.layers.Dense(c)
        self.r = tf.keras.layers.Activation('relu')
        self.s = tf.keras.layers.Activation('sigmoid')
        self.rs = tf.keras.layers.Reshape((1, 1, c))

    def call(self, x):
        y = self.a(x)
        y = self.d1(y)
        y = self.r(y)
        y = self.d2(y)
        y = self.s(y)
        y = self.rs(y)
        r = x * y
        return r
    

# m = tf.keras.Sequential([
#     tf.keras.Input(shape=(40, 40, 64)),
#     SE_Block(64)
# ])
# m.summary()
