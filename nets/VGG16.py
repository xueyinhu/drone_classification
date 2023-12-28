import tensorflow as tf

def VGG16(config):
    return tf.keras.applications.VGG16(
        input_shape=config.input_size,
        weights=None,
        classes=5
    )