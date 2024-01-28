import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class AE(Model):

    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Sequential([
            layers.Conv2D(16, (3, 3), (1, 1), 'same', activation='relu'),
            layers.MaxPool2D((4, 4), (4, 4), 'same'),
            layers.Conv2D(32, (3, 3), (1, 1), 'same', activation='relu'),
            layers.MaxPool2D((4, 4), (4, 4), 'same'),
            layers.Conv2D(64, (3, 3), (1, 1), 'same', activation='relu'),
            layers.MaxPool2D((4, 4), (4, 4), 'same'),
            layers.Conv2D(256, (5, 5), (1, 1), activation='relu'),
            layers.Flatten(),
            # layers.Dense(128, activation='relu'),
        ])
        self.decoder = Sequential([
            layers.Dense(256, activation='relu'),
            layers.Reshape((1, 1, 256)),
            layers.UpSampling2D((5, 5)),
            layers.Conv2D(64, (3, 3), (1, 1), 'same', activation='relu'),
            layers.UpSampling2D((4, 4)),
            layers.Conv2D(32, (3, 3), (1, 1), 'same', activation='relu'),
            layers.UpSampling2D((4, 4)),
            layers.Conv2D(16, (3, 3), (1, 1), 'same', activation='relu'),
            layers.UpSampling2D((4, 4)),
            layers.Conv2D(3, (3, 3), (1, 1), 'same', activation='relu'),
        ])

    def call(self, x):
        return self.decoder(
            self.encoder(
                x
            )
        )














