import tensorflow as tf

def MobileNetV2(config):
    return tf.keras.applications.MobileNetV2(
        input_shape=config.input_size,
        weights=None,
        classes=5
    )

