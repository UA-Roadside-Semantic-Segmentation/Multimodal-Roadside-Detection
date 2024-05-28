import numpy as np
import tensorflow as tf
import random
import h5py
import sys
from time import time
# import dl_lidar.dataset_viewer as dv

def data_gen(filename, ttv, num_classes, batch_size, augment):
    """
    :param ttv: testing, training, or validation
    """
    ttv_compare = ['testing', 'training', 'validation']
    if ttv not in ttv_compare:
        print('Error in data_gen: ttv must be exactly "testing", "training", or "validation"')
        sys.exit(1)

    hf = h5py.File(filename, 'r')
    data = hf[ttv + '_data']
    labels = hf[ttv + '_labeled']

    print(data.shape)
    print(labels.shape)

    while True:
        batch_indices = np.random.choice(data.shape[0], size=batch_size, replace=False)
        batch_indices.sort()  # ensures the random indices are in ascending order. limitation of h5

        batch_data = np.empty([batch_size, data.shape[1], data.shape[2], data.shape[3]])
        batch_labels = np.empty([batch_size, labels.shape[1], data.shape[2]])

        for c, i in enumerate(batch_indices):
            batch_data[c] = data[i]
            batch_labels[c] = labels[i]

        # visualize = dv.DatasetViewer(batch_data, batch_labels, 'test')
        # visualize.show()

        batch_labels_one_hot = tf.one_hot(batch_labels.astype(np.int32), num_classes)

        if augment:
            if random.choice([True, False]):
                batch_data = np.flip(batch_data, axis=2)
                batch_labels_one_hot = np.flip(batch_labels_one_hot, axis=2)

            noise = np.random.normal(0, .001, batch_data[0].shape)
            batch_data = batch_data + noise

            scale = np.random.normal(1.0, 0.1)
            batch_data[..., [0, 1, 2]] = batch_data[..., [0, 1, 2]] * scale

        yield batch_data, batch_labels_one_hot


""" Used to benchmark changes made to the data generator """

# batch = 100
# iterations = 5
# gen = data_gen('/home/mteastepp/Desktop/Lidar_Research/Data/16Line-dataset.hdf5', 'training', 2, batch, augment=False)
#
# start = time()
# for i in range(iterations):
#     print('Iteration: {}'.format(i))
#     [x, y] = next(gen)
#
#
# end = time()
# total = end - start
# print('Total Time for {} iterations of {} images: {}'.format(iterations, batch, total))
# print('Average time per iteration: {}'.format(total/iterations))
