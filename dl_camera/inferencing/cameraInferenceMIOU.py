import h5py
import tensorflow as tf
import numpy as np
import sys
import statistics
import cv2
import csv

class MeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # sparse code
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super(MeanIoU, self).update_state(y_true, y_pred, sample_weight)

if len(sys.argv) != 2:
    print("Usage: python3 {} infered_H5_File".format(sys.argv[0]))
    sys.exit(0)

inferred = h5py.File(sys.argv[1], 'r')

raw_data = inferred['img']
labeled_data = inferred['labeled']
inferred_data = inferred['inferred']


idx = 16
cv2.imwrite('lowLightInf.png', np.array(inferred_data[idx, :, :, 0] * 255).astype(np.uint8))
cv2.imwrite('lowLightLabel.png', labeled_data[idx, :, :, 0] * 255)
cv2.imwrite('lowLightRaw.png', (raw_data[idx]+1)*127.5)


results = list()

size = labeled_data.shape[0]
# size = 75
for idx, i in enumerate(labeled_data):
    miou = MeanIoU(2)
    miou.update_state(labeled_data[idx], inferred_data[idx])
    results.append(miou.result().numpy())
    print("{} of {}".format(idx, size))

miou.update_state(labeled_data, inferred_data)
print(miou.result().numpy())

avg = statistics.fmean(results)

variance = sum([((x-avg) ** 2) for x in results])/len(results)
stdDev = variance ** 0.5

print("Average: {}%".format(avg*100))
print("Standard Deviation: {}%".format(stdDev*100))

