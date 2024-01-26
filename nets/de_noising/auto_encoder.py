import tensorflow as tf
from tensorflow.python.keras import layers, Model, Sequential


class AE(Model):

    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Sequential([
            layers.Conv2D(64, (3, 3), (2, 2), 'same', activation='relu'),
            layers.Conv2D(256, (3, 3), (2, 2), 'same', activation='relu'),
            layers.Conv2D(512, (3, 3), (2, 2), 'same', activation='relu'),
            layers.Conv2D(1024, (3, 3), (2, 2), 'same', activation='relu'),
            layers.Conv2D(2048, (3, 3), (2, 2), 'same', activation='relu'),
            layers.Conv2D(4096, (3, 3), (7, 7), 'same', activation='relu'),
            layers.Flatten(),
            layers.Dense(1024, activation='relu'),
        ])
        self.decoder = Sequential([
            layers.Dense(4096, activation='relu'),
            layers.UpSampling2D((7, 7)),
            layers.Conv2D(2048, (3, 3), (1, 1), 'same', activation='relu'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(1024, (3, 3), (1, 1), 'same', activation='relu'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(512, (3, 3), (1, 1), 'same', activation='relu'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(256, (3, 3), (1, 1), 'same', activation='relu'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(64, (3, 3), (1, 1), 'same', activation='relu'),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(3, (3, 3), (1, 1), 'same', activation='relu'),
        ])

    def call(self, x):
        return self.decoder(
            self.encoder(
                x
            )
        )














