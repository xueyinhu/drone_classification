import os
import tensorflow as tf


def preprocess_image(imp):
    return tf.cast(
        tf.image.decode_jpeg(
            contents=tf.io.read_file(
                imp
            ),
            channels=3
        ),
        tf.float32
    ) / 255.


def load_im(dmn, lmn):
    return preprocess_image(dmn), preprocess_image(lmn)


def load_imp_list(inp='j:/dn_data/'):
    rdt = [inp + 'train/' + im for im in os.listdir(inp + 'train/')]
    rlb = [rd[:-6] + '00.jpg' for rd in rdt]
    edt = [inp + 'valid/' + im for im in os.listdir(inp + 'valid/')]
    elb = [ed[:-6] + '00.jpg' for ed in edt]
    return rdt, rlb, edt, elb


def gen_dataset(bs=64):
    rdt, rlb, edt, elb = load_imp_list()
    rds = tf.data.Dataset.from_tensor_slices((rdt, rlb))
    rds = rds.map(load_im)
    eds = tf.data.Dataset.from_tensor_slices((edt, elb))
    eds = eds.map(load_im)
    rds = rds.shuffle(buffer_size=len(rdt), reshuffle_each_iteration=True).repeat()
    rds = rds.batch(bs).prefetch(buffer_size=1)
    eds = eds.batch(bs).prefetch(buffer_size=1)
    return rds, eds


def gen_dataset_fit():
    rdt, rlb, edt, elb = load_imp_list()
    rdt = [preprocess_image(i) for i in rdt]
    rlb = [preprocess_image(i) for i in rlb]
    edt = [preprocess_image(i) for i in edt]
    elb = [preprocess_image(i) for i in elb]
    return rdt, rlb, edt, elb
