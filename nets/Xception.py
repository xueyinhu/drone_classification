import tensorflow as tf

def Xception(config):
    return tf.keras.applications.Xception(
        input_shape=config.input_size,
        weights=None,
        classes=5
    )