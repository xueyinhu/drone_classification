import tensorflow as tf

def InceptionV3(config):
    return tf.keras.applications.InceptionV3(
        input_shape=config.input_size,
        weights=None,
        classes=5
    )
