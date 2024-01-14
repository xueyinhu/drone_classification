import tensorflow as tf

def MobileNetV3Small(config):
    return tf.keras.applications.MobileNetV3Small(
        input_shape=config.input_size,
        weights=None,
        classes=5
    )
