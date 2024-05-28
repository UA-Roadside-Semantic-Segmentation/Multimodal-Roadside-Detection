import h5py
import dl_lidar.dataset_viewer as dv
import tensorflow as tf
import numpy as np
import sys

if len(sys.argv) != 4:
    print("Usage: python3 {} labeled_H5_File infered_H5_File scanlines".format(sys.argv[0]))
    sys.exit(0)

title = 'All Lidar Train Jetson Xavier Inference'

startIndex = 0

hf = h5py.File(sys.argv[1], 'r')
inferedHF = h5py.File(sys.argv[2], 'r')

range = hf['testing_data'].shape[0]
# range = 100
raw = hf['testing_data'][startIndex:startIndex+range]
infered = inferedHF['inferred'][startIndex:startIndex+range]
# infered = hf['testing_labeled'][startIndex:startIndex+range]


rawDV = dv.DatasetViewer(raw, tf.argmax(infered, -1), title, rows=int(sys.argv[3]))
# rawDV = dv.DatasetViewer(raw, infered, 'lowLight Testing Labeled', rows=int(sys.argv[3]))
rawDV.show()

hf.close()
inferedHF.close()
