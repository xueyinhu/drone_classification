compare_data = {
    "ECM-Net": [86.85,],
    "ResNet34": [84.89,],
    "InceptionV3": [82.02,],
    "Xception": [85.43,],
    "MobileNetV2": [77.19,],
    "ShuffleNetV2_0.5X": [81.72,],
    "ShuffleNetV2": [85.42,],

    "+SE": [86.57,],
    "+CBAM": [86.32,],
    "+ECA": [86.91,],
    "+P2A": [88.06,],
}


"""
Num GPUs Available:  1

********************   Load => Net   ******************** -  -  -  - [state: Over]

Found 68894 images belonging to 5 classes.
********************   Load => TrainImageDataGenerator   ******************** -  -  -  - [state: Over]

Found 7000 images belonging to 5 classes.
********************   Load => ValidImageDataGenerator   ******************** -  -  -  - [state: Over]

Epoch 1/80000
2152/2152 [==============================] - 1449s 647ms/step - loss: 1.1636 - accuracy: 0.5338 - val_loss: 0.7980 - val_accuracy: 0.7185 - lr: 0.0010
Epoch 2/80000
2152/2152 [==============================] - 1320s 613ms/step - loss: 0.7079 - accuracy: 0.7561 - val_loss: 0.7705 - val_accuracy: 0.7387 - lr: 0.0010
Epoch 3/80000
2152/2152 [==============================] - 1314s 610ms/step - loss: 0.5555 - accuracy: 0.8108 - val_loss: 0.7552 - val_accuracy: 0.7539 - lr: 0.0010
Epoch 4/80000
2152/2152 [==============================] - 1317s 612ms/step - loss: 0.4873 - accuracy: 0.8343 - val_loss: 0.5860 - val_accuracy: 0.8049 - lr: 0.0010
Epoch 5/80000
2152/2152 [==============================] - 1319s 613ms/step - loss: 0.4280 - accuracy: 0.8537 - val_loss: 0.6370 - val_accuracy: 0.7969 - lr: 0.0010
Epoch 6/80000
2152/2152 [==============================] - 1317s 612ms/step - loss: 0.3843 - accuracy: 0.8696 - val_loss: 0.5945 - val_accuracy: 0.8175 - lr: 0.0010
Epoch 7/80000
2152/2152 [==============================] - 1314s 610ms/step - loss: 0.3483 - accuracy: 0.8827 - val_loss: 0.6022 - val_accuracy: 0.8174 - lr: 0.0010
Epoch 8/80000
2152/2152 [==============================] - 1313s 610ms/step - loss: 0.1783 - accuracy: 0.9429 - val_loss: 0.4615 - val_accuracy: 0.8634 - lr: 1.0000e-04
Epoch 9/80000
2152/2152 [==============================] - 1313s 610ms/step - loss: 0.1124 - accuracy: 0.9651 - val_loss: 0.4622 - val_accuracy: 0.8678 - lr: 1.0000e-04
Epoch 10/80000
2152/2152 [==============================] - 1310s 609ms/step - loss: 0.0815 - accuracy: 0.9759 - val_loss: 0.4766 - val_accuracy: 0.8681 - lr: 1.0000e-04
Epoch 11/80000
2152/2152 [==============================] - 1316s 611ms/step - loss: 0.0627 - accuracy: 0.9814 - val_loss: 0.5098 - val_accuracy: 0.8677 - lr: 1.0000e-04
Epoch 12/80000
2152/2152 [==============================] - 1311s 609ms/step - loss: 0.0428 - accuracy: 0.9892 - val_loss: 0.5094 - val_accuracy: 0.8698 - lr: 1.0000e-05
Epoch 13/80000
2152/2152 [==============================] - 1310s 608ms/step - loss: 0.0396 - accuracy: 0.9898 - val_loss: 0.5137 - val_accuracy: 0.8685 - lr: 1.0000e-05
"""
