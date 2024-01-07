from absl import logging
import os

import tensorflow as tf

from config import get_config
from nets import *
from dataset_loader import get_idg
from utils import loader


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), '\n')

config = get_config()  

os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.log_level
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)

net = loader(CompareNetInc(config), 'Load => Net')

t_idg = loader(get_idg(config, mode=True), 'Load => TrainImageDataGenerator')
v_idg = loader(get_idg(config, mode=False), 'Load => ValidImageDataGenerator')

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
            config.save_path + 'model-' + config.net_name + '__epoch-{epoch:04d}__valLos-{val_loss:.4f}__valAcc-{val_accuracy:.2f}',
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

""" dropout=.3
Num GPUs Available:  1

********************   Load => Net   ******************** -  -  -  - [state: Over]

Found 68894 images belonging to 5 classes.
********************   Load => TrainImageDataGenerator   ******************** -  -  -  - [state: Over]

Found 7000 images belonging to 5 classes.
********************   Load => ValidImageDataGenerator   ******************** -  -  -  - [state: Over]

Epoch 1/80000
1076/1076 [==============================] - 1221s 1s/step - loss: 1.0462 - accuracy: 0.5930 - val_loss: 0.7528 - val_accuracy: 0.7348
Epoch 2/80000
1076/1076 [==============================] - 1109s 1s/step - loss: 0.5976 - accuracy: 0.7977 - val_loss: 0.6885 - val_accuracy: 0.7702
Epoch 3/80000
1076/1076 [==============================] - 1105s 1s/step - loss: 0.4670 - accuracy: 0.8432 - val_loss: 0.5728 - val_accuracy: 0.8125
Epoch 4/80000
1076/1076 [==============================] - 1105s 1s/step - loss: 0.3892 - accuracy: 0.8693 - val_loss: 0.5886 - val_accuracy: 0.8076
Epoch 5/80000
1076/1076 [==============================] - 1108s 1s/step - loss: 0.3313 - accuracy: 0.8879 - val_loss: 0.6121 - val_accuracy: 0.8168
Epoch 6/80000
1076/1076 [==============================] - 1126s 1s/step - loss: 0.2896 - accuracy: 0.9023 - val_loss: 0.6169 - val_accuracy: 0.8158
Epoch 7/80000
1076/1076 [==============================] - 1106s 1s/step - loss: 0.2589 - accuracy: 0.9138 - val_loss: 0.6690 - val_accuracy: 0.8131
Epoch 8/80000
1076/1076 [==============================] - 1106s 1s/step - loss: 0.2332 - accuracy: 0.9230 - val_loss: 0.6426 - val_accuracy: 0.8321
"""

""" dropout=.4
Num GPUs Available:  1 

********************   Load => Net   ******************** -  -  -  - [state: Over]

Found 68894 images belonging to 5 classes.
********************   Load => TrainImageDataGenerator   ******************** -  -  -  - [state: Over]

Found 7000 images belonging to 5 classes.
********************   Load => ValidImageDataGenerator   ******************** -  -  -  - [state: Over]

Epoch 1/80000
1076/1076 [==============================] - 1220s 1s/step - loss: 1.1269 - accuracy: 0.5519 - val_loss: 1.4534 - val_accuracy: 0.5235
Epoch 2/80000
1076/1076 [==============================] - 1110s 1s/step - loss: 0.6609 - accuracy: 0.7745 - val_loss: 0.6448 - val_accuracy: 0.7820
Epoch 3/80000
1076/1076 [==============================] - 1110s 1s/step - loss: 0.5170 - accuracy: 0.8240 - val_loss: 0.6277 - val_accuracy: 0.7967
Epoch 4/80000
1076/1076 [==============================] - 1111s 1s/step - loss: 0.4260 - accuracy: 0.8561 - val_loss: 1.9750 - val_accuracy: 0.5493
Epoch 5/80000
1076/1076 [==============================] - 1110s 1s/step - loss: 0.3701 - accuracy: 0.8747 - val_loss: 0.6037 - val_accuracy: 0.8155
Epoch 6/80000
1076/1076 [==============================] - 1112s 1s/step - loss: 0.3268 - accuracy: 0.8886 - val_loss: 0.6302 - val_accuracy: 0.8129
Epoch 7/80000
1076/1076 [==============================] - 1110s 1s/step - loss: 0.2936 - accuracy: 0.9010 - val_loss: 0.6307 - val_accuracy: 0.8149
Epoch 8/80000
1076/1076 [==============================] - 1109s 1s/step - loss: 0.2647 - accuracy: 0.9102 - val_loss: 0.6728 - val_accuracy: 0.8103
Epoch 9/80000
1076/1076 [==============================] - 1110s 1s/step - loss: 0.2418 - accuracy: 0.9190 - val_loss: 0.7356 - val_accuracy: 0.8108
Epoch 10/80000
1076/1076 [==============================] - 1113s 1s/step - loss: 0.2235 - accuracy: 0.9256 - val_loss: 0.8039 - val_accuracy: 0.8026
"""

""" lr,rate=.1
Num GPUs Available:  1

********************   Load => Net   ******************** -  -  -  - [state: Over]

Found 68894 images belonging to 5 classes.
********************   Load => TrainImageDataGenerator   ******************** -  -  -  - [state: Over]

Found 7000 images belonging to 5 classes.
********************   Load => ValidImageDataGenerator   ******************** -  -  -  - [state: Over]

Epoch 1/80000
1076/1076 [==============================] - 1228s 1s/step - loss: 1.0439 - accuracy: 0.5939 - val_loss: 1.3944 - val_accuracy: 0.5328 - lr: 0.0010
Epoch 2/80000
1076/1076 [==============================] - 1110s 1s/step - loss: 0.5953 - accuracy: 0.7984 - val_loss: 0.9216 - val_accuracy: 0.7048 - lr: 0.0010
Epoch 3/80000
1076/1076 [==============================] - 1112s 1s/step - loss: 0.4570 - accuracy: 0.8464 - val_loss: 0.6083 - val_accuracy: 0.8066 - lr: 0.0010
Epoch 4/80000
1076/1076 [==============================] - 1113s 1s/step - loss: 0.3738 - accuracy: 0.8749 - val_loss: 0.6051 - val_accuracy: 0.8098 - lr: 0.0010
Epoch 5/80000
1076/1076 [==============================] - 1110s 1s/step - loss: 0.3233 - accuracy: 0.8914 - val_loss: 0.7413 - val_accuracy: 0.7837 - lr: 0.0010
Epoch 6/80000
1076/1076 [==============================] - 1111s 1s/step - loss: 0.2827 - accuracy: 0.9044 - val_loss: 0.6846 - val_accuracy: 0.7963 - lr: 0.0010
Epoch 7/80000
1076/1076 [==============================] - 1111s 1s/step - loss: 0.2552 - accuracy: 0.9141 - val_loss: 0.7385 - val_accuracy: 0.8025 - lr: 0.0010
Epoch 8/80000
1076/1076 [==============================] - 1111s 1s/step - loss: 0.1298 - accuracy: 0.9595 - val_loss: 0.5170 - val_accuracy: 0.8594 - lr: 1.0000e-04
Epoch 9/80000
1076/1076 [==============================] - 1113s 1s/step - loss: 0.0783 - accuracy: 0.9775 - val_loss: 0.5338 - val_accuracy: 0.8601 - lr: 1.0000e-04
Epoch 10/80000
1076/1076 [==============================] - 1111s 1s/step - loss: 0.0554 - accuracy: 0.9848 - val_loss: 0.5607 - val_accuracy: 0.8610 - lr: 1.0000e-04
Epoch 11/80000
1076/1076 [==============================] - 1112s 1s/step - loss: 0.0416 - accuracy: 0.9891 - val_loss: 0.6118 - val_accuracy: 0.8587 - lr: 1.0000e-04
Epoch 12/80000
1076/1076 [==============================] - 1112s 1s/step - loss: 0.0292 - accuracy: 0.9930 - val_loss: 0.5879 - val_accuracy: 0.8627 - lr: 1.0000e-05
Epoch 13/80000
1076/1076 [==============================] - 1109s 1s/step - loss: 0.0284 - accuracy: 0.9935 - val_loss: 0.5931 - val_accuracy: 0.8617 - lr: 1.0000e-05
"""


""" Inception-V3
Num GPUs Available:  1

********************   Load => Net   ******************** -  -  -  - [state: Over]

Found 68894 images belonging to 5 classes.
********************   Load => TrainImageDataGenerator   ******************** -  -  -  - [state: Over]

Found 7000 images belonging to 5 classes.
********************   Load => ValidImageDataGenerator   ******************** -  -  -  - [state: Over]

Epoch 1/80000
2152/2152 [==============================] - 1609s 741ms/step - loss: 1.5679 - accuracy: 0.2629 - val_loss: 3.3557 - val_accuracy: 0.2021 - lr: 0.0010
Epoch 2/80000
2152/2152 [==============================] - 1542s 716ms/step - loss: 1.4100 - accuracy: 0.3394 - val_loss: 2.1078 - val_accuracy: 0.2692 - lr: 0.0010
Epoch 3/80000
2152/2152 [==============================] - 1538s 715ms/step - loss: 1.2677 - accuracy: 0.4248 - val_loss: 1.5971 - val_accuracy: 0.3273 - lr: 0.0010
Epoch 4/80000
2152/2152 [==============================] - 1535s 713ms/step - loss: 1.1242 - accuracy: 0.5069 - val_loss: 1.5782 - val_accuracy: 0.4044 - lr: 0.0010
Epoch 5/80000
2152/2152 [==============================] - 1532s 712ms/step - loss: 0.9483 - accuracy: 0.6153 - val_loss: 2.5021 - val_accuracy: 0.4321 - lr: 0.0010
Epoch 6/80000
2152/2152 [==============================] - 1533s 712ms/step - loss: 0.7580 - accuracy: 0.7101 - val_loss: 0.8962 - val_accuracy: 0.6560 - lr: 0.0010
Epoch 7/80000
2152/2152 [==============================] - 1532s 712ms/step - loss: 0.6261 - accuracy: 0.7636 - val_loss: 1.4597 - val_accuracy: 0.5674 - lr: 0.0010
Epoch 8/80000
2152/2152 [==============================] - 1532s 712ms/step - loss: 0.5490 - accuracy: 0.7906 - val_loss: 0.8219 - val_accuracy: 0.6846 - lr: 0.0010
Epoch 9/80000
2152/2152 [==============================] - 1533s 712ms/step - loss: 0.4893 - accuracy: 0.8128 - val_loss: 0.8715 - val_accuracy: 0.7064 - lr: 0.0010
Epoch 10/80000
2152/2152 [==============================] - 1535s 713ms/step - loss: 0.4461 - accuracy: 0.8294 - val_loss: 0.8856 - val_accuracy: 0.6977 - lr: 0.0010
Epoch 11/80000
2152/2152 [==============================] - 1544s 717ms/step - loss: 0.4018 - accuracy: 0.8456 - val_loss: 1.6065 - val_accuracy: 0.5011 - lr: 0.0010
Epoch 12/80000
2152/2152 [==============================] - 1553s 721ms/step - loss: 0.2466 - accuracy: 0.9061 - val_loss: 0.5828 - val_accuracy: 0.8158 - lr: 1.0000e-04
Epoch 13/80000
2152/2152 [==============================] - 1541s 716ms/step - loss: 0.1975 - accuracy: 0.9239 - val_loss: 0.6306 - val_accuracy: 0.8204 - lr: 1.0000e-04
Epoch 14/80000
2152/2152 [==============================] - 1545s 718ms/step - loss: 0.1705 - accuracy: 0.9338 - val_loss: 0.6990 - val_accuracy: 0.8210 - lr: 1.0000e-04
Epoch 15/80000
2152/2152 [==============================] - 1556s 723ms/step - loss: 0.1446 - accuracy: 0.9440 - val_loss: 0.7256 - val_accuracy: 0.8188 - lr: 1.0000e-04
Epoch 16/80000
2152/2152 [==============================] - 1558s 724ms/step - loss: 0.1161 - accuracy: 0.9568 - val_loss: 0.7574 - val_accuracy: 0.8214 - lr: 1.0000e-05
Epoch 17/80000
2152/2152 [==============================] - 1556s 723ms/step - loss: 0.1103 - accuracy: 0.9580 - val_loss: 0.7795 - val_accuracy: 0.8202 - lr: 1.0000e-05
"""

""" X
Num GPUs Available:  1 

********************   Load => Net   ******************** -  -  -  - [state: Over]

Found 68894 images belonging to 5 classes.
********************   Load => TrainImageDataGenerator   ******************** -  -  -  - [state: Over]

Found 7000 images belonging to 5 classes.
********************   Load => ValidImageDataGenerator   ******************** -  -  -  - [state: Over]

Epoch 1/80000
4305/4305 [==============================] - 2795s 648ms/step - loss: 1.6110 - accuracy: 0.2180 - val_loss: 2.9544 - val_accuracy: 0.2208 - lr: 0.0010
Epoch 2/80000
4305/4305 [==============================] - 2788s 648ms/step - loss: 1.5164 - accuracy: 0.3036 - val_loss: 1.4446 - val_accuracy: 0.3495 - lr: 0.0010
Epoch 3/80000
4305/4305 [==============================] - 2775s 644ms/step - loss: 1.2370 - accuracy: 0.4749 - val_loss: 1.3376 - val_accuracy: 0.4301 - lr: 0.0010
Epoch 4/80000
4305/4305 [==============================] - 2772s 644ms/step - loss: 0.8445 - accuracy: 0.6850 - val_loss: 0.7990 - val_accuracy: 0.7201 - lr: 0.0010
Epoch 5/80000
4305/4305 [==============================] - 2788s 648ms/step - loss: 0.5821 - accuracy: 0.7918 - val_loss: 2.5289 - val_accuracy: 0.3942 - lr: 0.0010
Epoch 6/80000
4305/4305 [==============================] - 2779s 646ms/step - loss: 0.4376 - accuracy: 0.8425 - val_loss: 0.6031 - val_accuracy: 0.7912 - lr: 0.0010
Epoch 7/80000
4305/4305 [==============================] - 2774s 644ms/step - loss: 0.3383 - accuracy: 0.8805 - val_loss: 1.0685 - val_accuracy: 0.6659 - lr: 0.0010
Epoch 8/80000
4305/4305 [==============================] - 2775s 645ms/step - loss: 0.2662 - accuracy: 0.9093 - val_loss: 1.6982 - val_accuracy: 0.5874 - lr: 0.0010
Epoch 9/80000
4305/4305 [==============================] - 2778s 645ms/step - loss: 0.2126 - accuracy: 0.9282 - val_loss: 0.8435 - val_accuracy: 0.7853 - lr: 0.0010
Epoch 10/80000
4305/4305 [==============================] - 2776s 645ms/step - loss: 0.0707 - accuracy: 0.9790 - val_loss: 0.6470 - val_accuracy: 0.8467 - lr: 1.0000e-04
Epoch 11/80000
4305/4305 [==============================] - 2809s 653ms/step - loss: 0.0279 - accuracy: 0.9921 - val_loss: 0.7092 - val_accuracy: 0.8543 - lr: 1.0000e-04
"""

""" M
Num GPUs Available:  1 

********************   Load => Net   ******************** -  -  -  - [state: Over]

Found 68894 images belonging to 5 classes.
********************   Load => TrainImageDataGenerator   ******************** -  -  -  - [state: Over]

Found 7000 images belonging to 5 classes.
********************   Load => ValidImageDataGenerator   ******************** -  -  -  - [state: Over]

Epoch 1/80000
2870/2870 [==============================] - 1196s 416ms/step - loss: 1.4469 - accuracy: 0.3716 - val_loss: 2.1044 - val_accuracy: 0.2020 - lr: 1.0000e-04
Epoch 2/80000
2870/2870 [==============================] - 1177s 410ms/step - loss: 1.1436 - accuracy: 0.5471 - val_loss: 6.3715 - val_accuracy: 0.2143 - lr: 1.0000e-04
Epoch 3/80000
2870/2870 [==============================] - 1176s 410ms/step - loss: 0.9507 - accuracy: 0.6310 - val_loss: 2.0587 - val_accuracy: 0.3678 - lr: 1.0000e-04
Epoch 4/80000
2870/2870 [==============================] - 1238s 431ms/step - loss: 0.8227 - accuracy: 0.6839 - val_loss: 2.5125 - val_accuracy: 0.3701 - lr: 1.0000e-04
Epoch 5/80000
2870/2870 [==============================] - 1194s 416ms/step - loss: 0.7296 - accuracy: 0.7225 - val_loss: 4.1578 - val_accuracy: 0.3613 - lr: 1.0000e-04
Epoch 6/80000
2870/2870 [==============================] - 1224s 426ms/step - loss: 0.6591 - accuracy: 0.7500 - val_loss: 2.1283 - val_accuracy: 0.4340 - lr: 1.0000e-04
Epoch 7/80000
2870/2870 [==============================] - 1206s 420ms/step - loss: 0.4347 - accuracy: 0.8383 - val_loss: 0.6998 - val_accuracy: 0.7572 - lr: 1.0000e-05
Epoch 8/80000
2870/2870 [==============================] - 1218s 424ms/step - loss: 0.3447 - accuracy: 0.8710 - val_loss: 0.6961 - val_accuracy: 0.7613 - lr: 1.0000e-05
Epoch 9/80000
2870/2870 [==============================] - 1286s 448ms/step - loss: 0.2900 - accuracy: 0.8934 - val_loss: 0.8181 - val_accuracy: 0.7433 - lr: 1.0000e-05
Epoch 10/80000
2870/2870 [==============================] - 1216s 423ms/step - loss: 0.2474 - accuracy: 0.9113 - val_loss: 0.7088 - val_accuracy: 0.7719 - lr: 1.0000e-05
Epoch 11/80000
2870/2870 [==============================] - 1206s 420ms/step - loss: 0.2051 - accuracy: 0.9299 - val_loss: 0.7926 - val_accuracy: 0.7719 - lr: 1.0000e-05
Epoch 12/80000
2870/2870 [==============================] - 1212s 422ms/step - loss: 0.1576 - accuracy: 0.9515 - val_loss: 0.7377 - val_accuracy: 0.7738 - lr: 1.0000e-06
Epoch 13/80000
2870/2870 [==============================] - 1196s 417ms/step - loss: 0.1509 - accuracy: 0.9541 - val_loss: 0.7440 - val_accuracy: 0.7719 - lr: 1.0000e-06
"""

""" Mo
Num GPUs Available:  1 

********************   Load => Net   ******************** -  -  -  - [state: Over]

Found 68894 images belonging to 5 classes.
********************   Load => TrainImageDataGenerator   ******************** -  -  -  - [state: Over]

Found 7000 images belonging to 5 classes.
********************   Load => ValidImageDataGenerator   ******************** -  -  -  - [state: Over]

Epoch 1/80000
2870/2870 [==============================] - 1178s 409ms/step - loss: 1.5299 - accuracy: 0.2872 - val_loss: 1.9440 - val_accuracy: 0.2010 - lr: 0.0010
Epoch 2/80000
2870/2870 [==============================] - 1184s 413ms/step - loss: 1.4302 - accuracy: 0.3435 - val_loss: 4.6530 - val_accuracy: 0.2192 - lr: 0.0010
Epoch 3/80000
2870/2870 [==============================] - 1176s 410ms/step - loss: 1.3498 - accuracy: 0.3852 - val_loss: 14.1494 - val_accuracy: 0.2009 - lr: 0.0010
Epoch 4/80000
2870/2870 [==============================] - 1184s 413ms/step - loss: 1.2692 - accuracy: 0.4347 - val_loss: 3.1304 - val_accuracy: 0.2713 - lr: 0.0010
Epoch 5/80000
2870/2870 [==============================] - 1182s 412ms/step - loss: 1.0375 - accuracy: 0.5639 - val_loss: 1.0718 - val_accuracy: 0.5540 - lr: 1.0000e-04
Epoch 6/80000
2870/2870 [==============================] - 1183s 412ms/step - loss: 0.9015 - accuracy: 0.6422 - val_loss: 0.9887 - val_accuracy: 0.6145 - lr: 1.0000e-04
Epoch 7/80000
2870/2870 [==============================] - 1179s 411ms/step - loss: 0.8118 - accuracy: 0.6839 - val_loss: 0.9758 - val_accuracy: 0.6342 - lr: 1.0000e-04
Epoch 8/80000
2870/2870 [==============================] - 1180s 411ms/step - loss: 0.7294 - accuracy: 0.7151 - val_loss: 0.8553 - val_accuracy: 0.6876 - lr: 1.0000e-04
Epoch 9/80000
2870/2870 [==============================] - 1173s 409ms/step - loss: 0.6596 - accuracy: 0.7370 - val_loss: 0.8293 - val_accuracy: 0.7005 - lr: 1.0000e-04
Epoch 10/80000
2870/2870 [==============================] - 1172s 408ms/step - loss: 0.6005 - accuracy: 0.7597 - val_loss: 0.9039 - val_accuracy: 0.6783 - lr: 1.0000e-04
Epoch 11/80000
2870/2870 [==============================] - 1174s 409ms/step - loss: 0.5499 - accuracy: 0.7789 - val_loss: 0.8793 - val_accuracy: 0.7159 - lr: 1.0000e-04
Epoch 12/80000
2870/2870 [==============================] - 1170s 408ms/step - loss: 0.5072 - accuracy: 0.7951 - val_loss: 0.8952 - val_accuracy: 0.7070 - lr: 1.0000e-04
Epoch 13/80000
2870/2870 [==============================] - 1173s 409ms/step - loss: 0.4261 - accuracy: 0.8283 - val_loss: 0.8280 - val_accuracy: 0.7355 - lr: 1.0000e-05
Epoch 14/80000
2870/2870 [==============================] - 1168s 407ms/step - loss: 0.4050 - accuracy: 0.8352 - val_loss: 0.8501 - val_accuracy: 0.7380 - lr: 1.0000e-05
Epoch 15/80000
2870/2870 [==============================] - 1170s 408ms/step - loss: 0.3948 - accuracy: 0.8384 - val_loss: 0.8675 - val_accuracy: 0.7383 - lr: 1.0000e-05
Epoch 16/80000
2870/2870 [==============================] - 1171s 408ms/step - loss: 0.3834 - accuracy: 0.8442 - val_loss: 0.8714 - val_accuracy: 0.7394 - lr: 1.0000e-05
Epoch 17/80000
2870/2870 [==============================] - 1170s 408ms/step - loss: 0.3715 - accuracy: 0.8494 - val_loss: 0.8873 - val_accuracy: 0.7394 - lr: 1.0000e-06
Epoch 18/80000
2870/2870 [==============================] - 1170s 408ms/step - loss: 0.3689 - accuracy: 0.8518 - val_loss: 0.8901 - val_accuracy: 0.7371 - lr: 1.0000e-06
"""


""" ResNet34
Num GPUs Available:  1

********************   Load => Net   ******************** -  -  -  - [state: Over]

Found 68894 images belonging to 5 classes.
********************   Load => TrainImageDataGenerator   ******************** -  -  -  - [state: Over]

Found 7000 images belonging to 5 classes.
********************   Load => ValidImageDataGenerator   ******************** -  -  -  - [state: Over]

Epoch 1/80000
2152/2152 [==============================] - 1290s 597ms/step - loss: 1.6014 - accuracy: 0.2402 - val_loss: 1.7524 - val_accuracy: 0.2332 - lr: 0.0010
Epoch 2/80000
2152/2152 [==============================] - 1276s 593ms/step - loss: 1.5045 - accuracy: 0.3167 - val_loss: 2.7189 - val_accuracy: 0.2091 - lr: 0.0010
Epoch 3/80000
2152/2152 [==============================] - 1273s 592ms/step - loss: 1.3552 - accuracy: 0.4019 - val_loss: 2.6232 - val_accuracy: 0.2466 - lr: 0.0010
Epoch 4/80000
2152/2152 [==============================] - 1377s 640ms/step - loss: 1.1702 - accuracy: 0.5068 - val_loss: 4.0262 - val_accuracy: 0.2772 - lr: 0.0010
Epoch 5/80000
2152/2152 [==============================] - 1301s 604ms/step - loss: 0.7565 - accuracy: 0.7196 - val_loss: 0.7555 - val_accuracy: 0.7317 - lr: 1.0000e-04
Epoch 6/80000
2152/2152 [==============================] - 1262s 586ms/step - loss: 0.5891 - accuracy: 0.7912 - val_loss: 0.6373 - val_accuracy: 0.7853 - lr: 1.0000e-04
Epoch 7/80000
2152/2152 [==============================] - 1254s 583ms/step - loss: 0.4909 - accuracy: 0.8261 - val_loss: 0.6427 - val_accuracy: 0.7873 - lr: 1.0000e-04
Epoch 8/80000
2152/2152 [==============================] - 1254s 582ms/step - loss: 0.4190 - accuracy: 0.8512 - val_loss: 0.5938 - val_accuracy: 0.8081 - lr: 1.0000e-04
Epoch 9/80000
2152/2152 [==============================] - 1247s 579ms/step - loss: 0.3622 - accuracy: 0.8699 - val_loss: 0.6899 - val_accuracy: 0.8015 - lr: 1.0000e-04
Epoch 10/80000
2152/2152 [==============================] - 1263s 587ms/step - loss: 0.3078 - accuracy: 0.8892 - val_loss: 0.5750 - val_accuracy: 0.8244 - lr: 1.0000e-04
Epoch 11/80000
2152/2152 [==============================] - 1262s 586ms/step - loss: 0.2594 - accuracy: 0.9063 - val_loss: 0.5429 - val_accuracy: 0.8386 - lr: 1.0000e-04
Epoch 12/80000
2152/2152 [==============================] - 1248s 580ms/step - loss: 0.2161 - accuracy: 0.9214 - val_loss: 0.6898 - val_accuracy: 0.8089 - lr: 1.0000e-04
Epoch 13/80000
2152/2152 [==============================] - 1248s 580ms/step - loss: 0.1795 - accuracy: 0.9359 - val_loss: 0.6858 - val_accuracy: 0.8149 - lr: 1.0000e-04
Epoch 14/80000
2152/2152 [==============================] - 1244s 578ms/step - loss: 0.1466 - accuracy: 0.9479 - val_loss: 0.8106 - val_accuracy: 0.8109 - lr: 1.0000e-04
Epoch 15/80000
2152/2152 [==============================] - 1249s 580ms/step - loss: 0.0862 - accuracy: 0.9721 - val_loss: 0.6660 - val_accuracy: 0.8475 - lr: 1.0000e-05
Epoch 16/80000
2152/2152 [==============================] - 1302s 605ms/step - loss: 0.0716 - accuracy: 0.9771 - val_loss: 0.6917 - val_accuracy: 0.8489 - lr: 1.0000e-05
"""

""" MyNet
Num GPUs Available:  1

********************   Load => Net   ******************** -  -  -  - [state: Over]

Found 68894 images belonging to 5 classes.
********************   Load => TrainImageDataGenerator   ******************** -  -  -  - [state: Over]

Found 7000 images belonging to 5 classes.
********************   Load => ValidImageDataGenerator   ******************** -  -  -  - [state: Over]

Epoch 1/80000
1076/1076 [==============================] - 1316s 1s/step - loss: 1.0450 - accuracy: 0.5921 - val_loss: 1.0159 - val_accuracy: 0.6458 - lr: 0.0010
Epoch 2/80000
1076/1076 [==============================] - 1175s 1s/step - loss: 0.5890 - accuracy: 0.8010 - val_loss: 0.6705 - val_accuracy: 0.7782 - lr: 0.0010
Epoch 3/80000
1076/1076 [==============================] - 1214s 1s/step - loss: 0.4533 - accuracy: 0.8463 - val_loss: 0.6128 - val_accuracy: 0.8010 - lr: 0.0010
Epoch 4/80000
1076/1076 [==============================] - 1192s 1s/step - loss: 0.3707 - accuracy: 0.8748 - val_loss: 0.5744 - val_accuracy: 0.8222 - lr: 0.0010
Epoch 5/80000
1076/1076 [==============================] - 1169s 1s/step - loss: 0.3133 - accuracy: 0.8937 - val_loss: 0.9703 - val_accuracy: 0.7618 - lr: 0.0010
Epoch 6/80000
1076/1076 [==============================] - 1158s 1s/step - loss: 0.2740 - accuracy: 0.9074 - val_loss: 0.7956 - val_accuracy: 0.7923 - lr: 0.0010
Epoch 7/80000
1076/1076 [==============================] - 1144s 1s/step - loss: 0.2509 - accuracy: 0.9151 - val_loss: 0.6258 - val_accuracy: 0.8161 - lr: 0.0010
Epoch 8/80000
1076/1076 [==============================] - 1146s 1s/step - loss: 0.1290 - accuracy: 0.9593 - val_loss: 0.4864 - val_accuracy: 0.8650 - lr: 1.0000e-04
Epoch 9/80000
1076/1076 [==============================] - 1125s 1s/step - loss: 0.0794 - accuracy: 0.9764 - val_loss: 0.4997 - val_accuracy: 0.8680 - lr: 1.0000e-04
Epoch 10/80000
1076/1076 [==============================] - 1174s 1s/step - loss: 0.0554 - accuracy: 0.9844 - val_loss: 0.5492 - val_accuracy: 0.8624 - lr: 1.0000e-04
Epoch 11/80000
1076/1076 [==============================] - 1139s 1s/step - loss: 0.0428 - accuracy: 0.9880 - val_loss: 0.5952 - val_accuracy: 0.8594 - lr: 1.0000e-04
Epoch 12/80000
1076/1076 [==============================] - 1118s 1s/step - loss: 0.0294 - accuracy: 0.9931 - val_loss: 0.5599 - val_accuracy: 0.8661 - lr: 1.0000e-05
Epoch 13/80000
1076/1076 [==============================] - 1119s 1s/step - loss: 0.0283 - accuracy: 0.9934 - val_loss: 0.5629 - val_accuracy: 0.8667 - lr: 1.0000e-05
"""

""" ShuffleNetV2
Num GPUs Available:  1

********************   Load => Net   ******************** -  -  -  - [state: Over]

Found 68894 images belonging to 5 classes.
********************   Load => TrainImageDataGenerator   ******************** -  -  -  - [state: Over]

Found 7000 images belonging to 5 classes.
********************   Load => ValidImageDataGenerator   ******************** -  -  -  - [state: Over]

Epoch 1/80000
1076/1076 [==============================] - 1448s 1s/step - loss: 1.7327 - accuracy: 0.2684 - val_loss: 3.0633 - val_accuracy: 0.2222 - lr: 0.0010
Epoch 2/80000
1076/1076 [==============================] - 823s 764ms/step - loss: 1.3379 - accuracy: 0.4368 - val_loss: 1.4238 - val_accuracy: 0.4586 - lr: 0.0010
Epoch 3/80000
1076/1076 [==============================] - 996s 926ms/step - loss: 0.9866 - accuracy: 0.6178 - val_loss: 1.1730 - val_accuracy: 0.5442 - lr: 0.0010
Epoch 4/80000
1076/1076 [==============================] - 826s 767ms/step - loss: 0.7668 - accuracy: 0.7144 - val_loss: 1.2311 - val_accuracy: 0.5549 - lr: 0.0010
Epoch 5/80000
1076/1076 [==============================] - 826s 767ms/step - loss: 0.6437 - accuracy: 0.7620 - val_loss: 1.5304 - val_accuracy: 0.5186 - lr: 0.0010
Epoch 6/80000
1076/1076 [==============================] - 816s 758ms/step - loss: 0.5556 - accuracy: 0.7994 - val_loss: 5.3650 - val_accuracy: 0.3238 - lr: 0.0010
Epoch 7/80000
1076/1076 [==============================] - 819s 761ms/step - loss: 0.2968 - accuracy: 0.8986 - val_loss: 0.6782 - val_accuracy: 0.7804 - lr: 1.0000e-04
Epoch 8/80000
1076/1076 [==============================] - 833s 774ms/step - loss: 0.1558 - accuracy: 0.9524 - val_loss: 0.7700 - val_accuracy: 0.7848 - lr: 1.0000e-04
Epoch 9/80000
1076/1076 [==============================] - 824s 766ms/step - loss: 0.0691 - accuracy: 0.9861 - val_loss: 0.5749 - val_accuracy: 0.8407 - lr: 1.0000e-04
Epoch 10/80000
1076/1076 [==============================] - 823s 764ms/step - loss: 0.0267 - accuracy: 0.9973 - val_loss: 0.6059 - val_accuracy: 0.8473 - lr: 1.0000e-04
Epoch 11/80000
1076/1076 [==============================] - 809s 752ms/step - loss: 0.0178 - accuracy: 0.9984 - val_loss: 0.8012 - val_accuracy: 0.8243 - lr: 1.0000e-04
Epoch 12/80000
1076/1076 [==============================] - 807s 750ms/step - loss: 0.0158 - accuracy: 0.9983 - val_loss: 0.7773 - val_accuracy: 0.8350 - lr: 1.0000e-04
Epoch 13/80000
1076/1076 [==============================] - 815s 757ms/step - loss: 0.0076 - accuracy: 0.9993 - val_loss: 0.6672 - val_accuracy: 0.8545 - lr: 1.0000e-05
Epoch 14/80000
1076/1076 [==============================] - 821s 763ms/step - loss: 0.0037 - accuracy: 0.9999 - val_loss: 0.6732 - val_accuracy: 0.8542 - lr: 1.0000e-05
"""
