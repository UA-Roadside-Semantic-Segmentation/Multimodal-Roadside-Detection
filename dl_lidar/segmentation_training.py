import tensorflow as tf
import dl_lidar.data_gen as data_gen
from time import time
#from deeplab.model_pointclouds import Deeplabv3
import cv2
import numpy as np
import dl_lidar.create_model as create_model

# from dataset_viewer import DatasetViewer

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#import keras2onnx
#import pandas as pd
#from dataset_viewer import DatasetViewer

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath='OS2-Truck-Weights.h5', monitor='val_mean_io_u',
                                    mode='max', save_best_only=True,
                                    save_weights_only=True),
    tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()), update_freq='batch')
]


class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # sparse code
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(MeanIoU, self).update_state(y_true, y_pred, sample_weight)


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    # change number of scan lines here
    scan_lines = 64
    model = create_model.create_model(scan_lines)

    print(model.count_params())

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy', MeanIoU(2)])

    model.load_weights('All16PCWeights.h5')

    test_gen = data_gen.data_gen('os2-truck-dataset.hdf5', 'testing', 2, 64, augment=False)
    next(test_gen)

    history = model.fit(x=data_gen.data_gen('os2-truck-dataset.hdf5', 'training', 2, 32, augment=True),
              validation_data=data_gen.data_gen('os2-truck-dataset.hdf5', 'validation', 2, 64, augment=False),
              epochs=1000, steps_per_epoch=1344//32, validation_steps=3, callbacks=callbacks)


# val_gen = data_gen.data_gen('16Line-dataset.hdf5', 'testing', 2, 512, augment=False)
#
#
# data = next(val_gen)
# result = model(data)
# #print(data[0].shape)
#
# #print('done')
# #data[1][:, :, :, 1]
# dv = DatasetViewer(data[0], tf.argmax(result, -1), 'dataset_name', 16)
# dv.show()
#
# #cv2.imwrite('result_image.png', np.array((data[0][0, :, :, :]+1)*127.5).astype(np.int))
#cv2.imwrite('result_true.png', np.array(data[1][0, :, :, 0]*255).astype(np.int))
#cv2.imwrite('result.png', np.array(result[0, :, :, 0]*255).astype(np.int))

