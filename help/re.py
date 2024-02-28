from config_limits_2 import *

import tensorflow as tf

config = get_config()

def get_idg(config, mode=True):
    p = config.data_path
    b = config.batch_size
    t = config.data_image_size
    if mode:
        d = config.data_idg_train_cfg
    else:
        d = config.data_idg_valid_cfg
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    ).flow_from_directory(
        directory=p + d, 
        batch_size=b, 
        target_size=t, 
        shuffle=True,
    )

net = tf.keras.applications.ResNet50(
    input_shape=config.input_size,
    weights=None,
    classes=5
)

t_idg = get_idg(config, mode=True)
v_idg = get_idg(config, mode=False)

net.compile(
    optimizer=config.optimizer, 
    loss=tf.keras.losses.get(config.loss_fn), 
    metrics=config.metrics
)

net.fit(
    t_idg, 
    steps_per_epoch=int(config.train_images_num / config.batch_size), 
    validation_data=v_idg, 
    validation_steps=int(config.valid_images_num / config.batch_size),
    epochs=config.epoch_count,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            config.save_path + 'model-' + config.net_name + '__epoch-{epoch:04d}__valLos-{val_loss:.4f}__valAcc-{val_accuracy:.4f}',
            save_weights_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=config.stop_patience
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=config.rdlr_rate,
            patience=config.rdlr_patience
        )
    ]
)

