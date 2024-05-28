import tensorflow as tf
import h5py
import numpy as np
from dl_camera.deeplab.model import Deeplabv3
# from dl_camera.image_data_gen import image_data_generator
from dl_camera.inferencing.inferencing_image_data_gen import image_data_generator
import sys
import cv2
from time import time
import math
import statistics

# if len(sys.argv) != 4:
#     print("Usage: python3 {} dataset weights_file inferred_file_output".format(sys.argv[0]))
#     sys.exit(0)

class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # sparse code
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(MeanIoU, self).update_state(y_true, y_pred, sample_weight)

height = 768
width = 1024

model = Deeplabv3(weights=None, input_shape=(height, width, 3), classes=2, backbone='mobilenetv2', alpha=1, activation='softmax')

# weightFile = sys.argv[2]
weightFile = '../reportSets/weightFiles/cameraTruckWeights.h5'
model.load_weights(weightFile)

# imgGen = image_data_generator(sys.argv[1], 'testing', 309, (height,width), False)
imgGen = image_data_generator('../reportSets/h5Files/cameraTruck.hdf5', 'testing', 1, (height,width), False)
[data, label] = next(imgGen)

print(data.shape)

for i in range(5):
    print('Inferring on dummy data to fully load model')
    result = model(np.expand_dims(data[0], 0))

time_sum = float()
start = time()
result = model(np.expand_dims(data[0], 0))
end = time()

time_sum += end - start

print('calculating MIOU')
miou_results = list()
miou = MeanIoU(2)
miou.update_state(label[0], result)
miou_results.append(miou.result().numpy())

# hfInferred = h5py.File(sys.argv[3], 'a')
hfInferred = h5py.File("../reportSets/h5Files/inferenceResults/cameraTruckTestSet.hdf5", 'a')

dsetInferred = hfInferred.create_dataset('inferred', maxshape=(None, height, width, 2), shape=(1, height, width, 2), compression='gzip')
dsetInferred[:] = result

dsetLabeled = hfInferred.create_dataset('labeled', maxshape=(None, height, width, 2), shape=(1, height, width, 2), compression='gzip')
dsetLabeled[:] = label[0]

dsetImg = hfInferred.create_dataset('img', maxshape=(None, height, width, 3), shape=(1, height, width, 3), compression='gzip')
dsetImg[:] = data[0]

print("inferered: {}".format(dsetInferred.shape))
print('labeled: {}'.format(dsetLabeled.shape))
print('img: {}'.format(dsetImg.shape))

iter = 308
for i in range(1, iter):
    [data, label] = next(imgGen)
    start = time()
    result = model(np.expand_dims(data[0], 0))
    end = time()
    time_sum += end - start

    miou.update_state(label[0], result)
    miou_results.append(miou.result().numpy())
    print(miou.result().numpy() * 100)

    dsetInferred.resize(dsetInferred.shape[0] + 1, axis=0)
    dsetInferred[-1] = result

    dsetLabeled.resize(dsetLabeled.shape[0] + 1, axis=0)
    dsetLabeled[-1] = label[0]

    dsetImg.resize(dsetImg.shape[0] + 1, axis=0)
    dsetImg[-1] = data[0]
    print("inferered: {}".format(dsetInferred.shape))
    print('labeled: {}'.format(dsetLabeled.shape))
    print('img: {}'.format(dsetImg.shape))

print('Calculating Average MIOU')
miouAvg = sum(miou_results) / len(miou_results)
print('Calculating Standard Deviation')
variance = sum([((x-miouAvg) ** 2) for x in miou_results])/len(miou_results)
stdDev = variance ** 0.5

print('MIOU Data for {}'.format(weightFile))
print('-----------------------------')
print('Average Inference Time: {} s'.format(time_sum/(len(miou_results))))
print("Average: {}%".format(miouAvg*100))
print("Standard Deviation: {}%".format(stdDev*100))
