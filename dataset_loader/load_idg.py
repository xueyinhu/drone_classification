import tensorflow as tf


idg = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)


def get_idg(config, mode=True):
    p = config.data_path
    b = config.batch_size
    t = config.data_image_size
    if mode:
        d = config.data_idg_train_cfg
    else:
        d = config.data_idg_valid_cfg
    return idg.flow_from_directory(
        directory=p + d, 
        batch_size=b, 
        target_size=t, 
        shuffle=True,
    )










