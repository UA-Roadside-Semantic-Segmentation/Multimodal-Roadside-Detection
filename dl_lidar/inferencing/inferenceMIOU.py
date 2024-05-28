import h5py
import tensorflow as tf
import numpy as np
import sys
# import dl_lidar.dataset_viewer as dv
import statistics

class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # sparse code
        # y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(MeanIoU, self).update_state(y_true, y_pred, sample_weight)

if len(sys.argv) != 3:
    print("Usage: python3 {} labeled_H5_File infered_H5_File".format(sys.argv[0]))
    sys.exit(0)

dataset = h5py.File(sys.argv[1], 'r')
inferred = h5py.File(sys.argv[2], 'r')

# dataset = h5py.File('../reportSets/h5Files/hiResTrain.hdf5', 'r')
# inferred = h5py.File('../reportSets/h5Files/hiResValidationCheck.hdf5', 'r')

raw_data = dataset['testing_data']
labeled_data = dataset['testing_labeled']
inferred_data = inferred['inferred']

# inferred_data = tf.argmax(inferred_data, -1)

# rawDV = dv.DatasetViewer(raw_data, tf.argmax(inferred_data, -1), 'OS2 Truck Inference', rows=64)
# rawDV.show()

# labeledDV =  dv.DatasetViewer(raw_data, labeled_data, "OS2-Labeled", rows=64)
# labeledDV.show()

results = list()

for c, i in enumerate(labeled_data):
    print("Calculating MIOU for {} of {}".format(c, len(labeled_data)))
    miou = MeanIoU(2)
    miou.update_state(labeled_data[c], inferred_data[c])
    results.append(miou.result().numpy())

print('Calculating Average MIOU')
avg = statistics.fmean(results)

print('Calculating StdDev')
variance = sum([((x-avg) ** 2) for x in results])/len(results)
stdDev = variance ** 0.5

print('Calculating MIOU of the whole set')
temp = MeanIoU(num_classes=2)
temp.update_state(labeled_data, inferred_data)

print('MIOU Data for {}'.format(sys.argv[2]))
print('-----------------------------')
print("Set MIOU: {}%".format(temp.result().numpy()*100))
print("Average: {}%".format(avg*100))
print("Standard Deviation: {}%".format(stdDev*100))

