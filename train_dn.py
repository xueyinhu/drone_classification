import tensorflow as tf

from nets import AE
from dataset_loader import gen_dataset_fit


def train_ae():
    net = AE()
    i = tf.keras.Input((320, 320, 3))
    t = tf.keras.Model(i, net(i))
    t.summary()
    rdt, rlb, edt, elb = gen_dataset_fit()
    net.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    net.fit(rdt, rlb, epochs=30, shuffle=True, validation_data=(edt, elb))


train_ae()
