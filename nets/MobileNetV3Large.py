import tensorflow as tf

def MobileNetV3Large(config):
    return tf.keras.applications.MobileNetV3Large(
        input_shape=config.input_size,
        weights=None,
        classes=5
    )
