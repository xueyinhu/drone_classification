import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense


def EasyCNNs(config):
    return Sequential([
        Conv2D(32, (5, 5), padding='same'),
        MaxPool2D(), # 160
        Conv2D(64, (3, 3), padding='same'),
        MaxPool2D(), # 80
        Conv2D(128, (3, 3), padding='same'),
        MaxPool2D(), # 40
        Conv2D(256, (3, 3), padding='same'),
        MaxPool2D(), # 20
        Conv2D(512, (3, 3), padding='same'),
        MaxPool2D(), # 10
        Conv2D(512, (3, 3)),
        MaxPool2D(), # 4
        Conv2D(1024, (3, 3)),
        MaxPool2D(), # 1
        Flatten(),
        Dense(5)
    ])

