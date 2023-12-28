import tensorflow as tf


def get_lr(init_lr, lr_steps, r):
    lr = [init_lr]
    for _ in range(len(lr_steps)):
        lr.append(lr[-1] * r)
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_steps, values=lr
    )

