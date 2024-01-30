import tensorflow as tf
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Flatten, Dense


class CB(Model):
    def __init__(self, c, a=2, b=2):
        super(self).__init__()
        self.b1 = Sequential([
            Conv2D(c, (1, 1)),
            MaxPool2D((3, 3), (a, b))
        ])
        self.b2 = Sequential([
            Conv2D(c, (1, 1)),
            Conv2D(c, (3, 3), (a, b), 'same', groups=c)
        ])
        self.b3 = Sequential([
            Conv2D(c, (1, 1)),
            Conv2D(c, (3, 3), (1, 1), 'same', groups=c),
            Conv2D(c, (3, 3), (a, b), 'same', groups=c)
        ])
        self.be = Conv2D(2 * c, (1, 1))

    def call(self, x):
        return self.be(
            tf.concat([
                self.b1(x),
                self.b2(x),
                self.b3(x)
            ], axis=-1)
        )

class TryModel_1(Model):
    def __init__(self):
        super(Inc1, self)._init__()
        self.head = Sequential([
            Conv2D(4, (3, 3), (1, 1), 'same')
        ])
        self.body = Sequential([
            CB(8),
            CB(16),
            CB(32, 3, 1),
            CB(64),
            CB(128),
            CB(256),
        ])
        self.tail = Sequential([
            Conv2D(512, (5, 5)),
            Flatten(),
            Dense(5)
        ])

    def call(self, x):
        return self.tail(
            self.body(
                self.head(
                    x
                )
            )
        )



