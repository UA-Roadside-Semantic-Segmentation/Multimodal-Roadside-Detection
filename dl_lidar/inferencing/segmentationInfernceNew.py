import tensorflow as tf
import h5py
import numpy as np
import dl_lidar.create_model as create_model
import sys
from time import time

class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # sparse code
        # y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(MeanIoU, self).update_state(y_true, y_pred, sample_weight)

# if len(sys.argv) != 5:
#     print("Usage: python3 {} dataset weights_file inferred_file_output scanlines".format(sys.argv[0]))
#     sys.exit(0)

# scanlines = int(sys.argv[4])
scanlines = 16
model = create_model.create_model(scanlines)

# weightFile = sys.argv[2]
weightFile = '../reportTraining/weightFiles/16-lineTruckWeights.h5'
model.load_weights(weightFile)



# hf = h5py.File(sys.argv[1], 'r')
hf = h5py.File('../reportTraining/H5Files/16-LineTruck.hdf5', 'r')
data = hf['testing_data']
labels = hf['testing_labeled']
size = data.shape[0]
raw = np.empty((1,scanlines,2048,4))

for i in range(0, 5):
    raw[0] = data[i]
    print("inferring on dummy data to fully load model")
    print(raw.shape)
    temp = model(tf.cast(raw, tf.float32))

raw[0] = data[0]

timeSum = 0
miouResults = list()

start = time()
results = model(tf.cast(raw, tf.float32))
end = time()
timeSum += end - start

print("Calculating MIOU for 0 of {}".format(size))
miou = MeanIoU(2)
miou.update_state(labels[0], results)
miouResults.append(miou.result().numpy())
print(miou.result().numpy()*100)

# hfInferred = h5py.File(sys.argv[3], 'a')
hfInferred = h5py.File('../reportTraining/H5Files/inferenceFiles/truck/16-lineTruckTestSet.hdf5', 'a')
dsetInferred = hfInferred.create_dataset('inferred', maxshape=((None,)+ results[0].shape), shape=((1,) + results[0].shape))
dsetInferred[:] = results

print(dsetInferred.shape)

iter = size
for i in range(1, iter):
    raw[0] = data[i]

    start = time()
    results = model(tf.cast(raw, tf.float32))
    end = time()
    timeSum += end - start

    print("Calculating MIOU for {} of {}".format(i, iter))
    miou = MeanIoU(2)
    miou.update_state(labels[i], results)
    miouResults.append(miou.result().numpy())
    print(miou.result().numpy() * 100)

    dsetInferred.resize(dsetInferred.shape[0] + 1, axis = 0)
    dsetInferred[-1] = results
    print(dsetInferred.shape)

print('Calculating Average MIOU')
avg = sum(miouResults)/len(miouResults)

print('Calculating StdDev')
variance = sum([((x-avg) ** 2) for x in miouResults])/len(miouResults)
stdDev = variance ** 0.5

print('MIOU Data for {}'.format(weightFile))
print('-----------------------------')
print("Average: {}%".format(avg*100))
print("Standard Deviation: {}%".format(stdDev*100))

print('inference time: {} s'.format(timeSum/size))

# dv = dataset_viewer.DatasetViewer(raw, tf.argmax(results, -1), 'test')
# dv.show()



