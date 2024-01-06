from absl import logging
import os

import tensorflow as tf

from config import get_config
from nets import MyNet, MobileNetV2, InceptionV3, ResNet50
from load_lr import get_lr
from dataset_loader import get_idg
from utils import ProgressBar, loader


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), '\n')

config = get_config()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.log_level
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)

# <->
net = loader(InceptionV3(config), 'Load => Net')
# net = loader(MobileNetV2(config), 'Load => Net')
# net = loader(MyNet(config), 'Load => Net')

t_idg = loader(get_idg(config, mode=True), 'Load => TrainImageDataGenerator')
v_idg = loader(get_idg(config, mode=False), 'Load => ValidImageDataGenerator')

lr = get_lr(config.init_lr, config.lr_steps, config.lr_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

cpt=tf.train.Checkpoint(
    step=tf.Variable(0, name='step'),
    optimizer=optimizer,
    model=net
)
cpt_m = tf.train.CheckpointManager(
    checkpoint=cpt,
    directory='./checkpoints/',
    max_to_keep=5
)
if cpt_m.latest_checkpoint:
    cpt.restore(cpt_m.latest_checkpoint)
    print('[*] load ckpt from {} at step {}.'.format(cpt_m.latest_checkpoint, cpt.step.numpy()))

prog_bar = ProgressBar(config.epoch_count, cpt.step.numpy())


tls = tf.keras.metrics.Mean()
tac = tf.keras.metrics.CategoricalAccuracy()
vls = tf.keras.metrics.Mean()
vac = tf.keras.metrics.CategoricalAccuracy()


@tf.function
def train_step(igs, lbs):
    with tf.GradientTape() as tape:
        pes = net(igs)
        loss = loss_fn(lbs, pes)
    gradients = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
    tls(loss)
    tac(lbs, pes)


@tf.function
def valid_step(igs, lbs):
    pes = net(igs)
    loss = loss_fn(lbs, pes)
    vls(loss)
    vac(lbs, pes)


print("\n [*] training start!")
for i in range(config.epoch_count):
    cpt.step.assign_add(1)
    steps = cpt.step.numpy()
    tls.reset_states()
    tac.reset_states()
    vls.reset_states()
    vac.reset_states()
    igs, lbs = t_idg.next()
    train_step(igs, lbs)
    igs, lbs = v_idg.next()
    valid_step(igs, lbs)
    prog_bar.update(
        "train_acc={:.4f}, train_loss={:.4f}, valid_acc={:.4f}, valid_loss={:.4f}".format(
            tac.result(), tls.result(), vac.result(), vls.result()
        )
    )
    if i != 0 and i % config.save_steps == 0:
        cpt_m.save()
        print("\n[*] save ckpt file at {}".format(cpt_m.latest_checkpoint))
print("\n [*] training done!")




