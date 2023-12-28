import tensorflow as tf

def ResNet50(config):
    return tf.keras.applications.ResNet50(
        input_shape=config.input_size,
        weights=None,
        classes=5
    )
