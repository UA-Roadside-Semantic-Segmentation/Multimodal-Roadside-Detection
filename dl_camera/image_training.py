import tensorflow as tf

from dl_camera.deeplab.model import Deeplabv3

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from dl_camera.image_data_gen import image_data_generator
from time import time
import cv2
import numpy as np
import math
from tensorflow.python.compiler.tensorrt import trt_convert as trt

#import keras2onnx
#import onnxruntime
#import tensorrt as trt


class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # sparse code
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(MeanIoU, self).update_state(y_true, y_pred, sample_weight)


callbacks = [
    ModelCheckpoint(filepath='blackflyTruckWeightsFull.h5', monitor='val_mean_io_u',
                                    mode='max', save_best_only=True, save_weights_only=True),
    TensorBoard(log_dir="logs/{}".format(time()), update_freq='batch', batch_size=32,)
]

# change height and width variables to match images
height = 768
width = 1024

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = Deeplabv3(weights=None, input_shape=(height, width, 3), classes=2,
                      backbone='mobilenetv2', alpha=1, activation='softmax')

    model.compile(optimizer=Adam(lr=0.0001, decay=1e-4), loss='categorical_crossentropy', metrics=['accuracy', MeanIoU(2)])

    print(model.count_params())

    model.load_weights('blackflyWeights.h5')

    history = model.fit(image_data_generator('blackfly-truck-dataset-full.hdf5', 'training', 16, (height, width), augment=True),
                       validation_data=image_data_generator('blackfly-truck-dataset-full.hdf5', 'validation', 20, (height, width), augment=False),
                       validation_steps=306//20, steps_per_epoch=2450//16, epochs=5000, callbacks=callbacks)

    # val_gen = image_data_generator('blackfly-trucks-dataset.hdf5', 'testing', 1, (600, 2048), augment=False)
    # # input = cv2.imread('blackfly_014_00290.jpg')
    #
    # # middle_y = math.floor(input.shape[0] / 2) - 300
    # # width_x = math.floor(input.shape[1] / 2)
    # #
    # # y1 = int(middle_y - 300)
    # # y2 = int(middle_y + 300)
    # #
    # # x1 = int(width_x - 1024)
    # # x2 = int(width_x + 1024)
    # #
    # # input = input[y1:y2, x1:x2]
    #
    # data = next(val_gen)
    #
    # result = model(data[0])
    #
    # cv2.imwrite('truck4.jpg', (data[0][0] + 1) * 127.5)
    # cv2.imwrite('truck_seg4.png', np.array(result[0, :, :, 0] * 255).astype(np.uint8))
