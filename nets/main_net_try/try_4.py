import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, ELU


class CB(Model):
    def __init__(self, c, a=2, b=2):
        super().__init__()
        self.b1 = Sequential([
            Conv2D(c, (1, 1)),
            BatchNormalization(),
            ELU(),
            MaxPool2D((3, 3), (a, b), 'same')
        ])
        self.b2 = Sequential([
            Conv2D(c, (1, 1)),
            BatchNormalization(),
            ELU(),
            Conv2D(c, (3, 3), (a, b), 'same', groups=c),
            BatchNormalization(),
            ELU(),
        ])
        self.b3 = Sequential([
            Conv2D(c, (1, 1)),
            BatchNormalization(),
            ELU(),
            Conv2D(c, (3, 3), (1, 1), 'same', groups=c),
            BatchNormalization(),
            ELU(),
            Conv2D(c, (3, 3), (a, b), 'same', groups=c),
            BatchNormalization(),
            ELU(),
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

class TryModel_4(Model):
    def __init__(self, configs):
        super().__init__()
        self.head = Sequential([
            Conv2D(8, (3, 3), (1, 1), 'same'),
            BatchNormalization(),
            ELU(),
        ])
        self.body = Sequential([
            CB(16),
            CB(32),
            CB(64, 3, 1),
            CB(128),
            CB(256),
            CB(512),
        ])
        self.tail = Sequential([
            Conv2D(1024, (5, 5)),
            BatchNormalization(),
            ELU(),
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