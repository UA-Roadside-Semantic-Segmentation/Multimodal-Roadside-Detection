import dataset
from time import time
import numpy as np


def shuffle_in_unison(raw, labeled):
    combined = list(zip(raw, labeled))
    np.random.shuffle(combined)
    raw, labeled= zip(*combined)
    return raw, labeled


def packagePCDDataset(raw, labeled, key, output, scanlines):
    output += '.hdf5'
    dataset.add_pcd_dataset_hdf5(raw, output, key + '_data', rows=scanlines, channels=4)
    dataset.add_pcd_dataset_hdf5(labeled, output, key + '_labeled', rows=scanlines, channels=5)
    return


def packageImgDataset(raw, labeled, key, output):
    output += '.hdf5'
    dataset.add_img_dataset_hdf5(raw, output, key + '_data')
    dataset.add_txt_dataset_hdf5(labeled, output, key + '_labeled')
    return


def createPCDH5(raw, labeled, output, scanlines, split):
    start = time()
    index1 = int(len(labeled) * split[0] / 100.0)
    index2 = int(len(labeled) * (split[0] + split[1]) / 100.0)

    raw, labeled = shuffle_in_unison(raw, labeled)

    lab_train, lab_val, lab_test = np.split(labeled, [index1, index2])
    raw_train, raw_val, raw_test = np.split(raw, [index1, index2])

    train_start = time()

    packagePCDDataset(raw_train.tolist(), lab_train.tolist(), 'training', output, scanlines)

    train_end = time()
    val_start = time()

    packagePCDDataset(raw_val.tolist(), lab_val.tolist(), 'validation', output, scanlines)

    val_end = time()
    test_start = time()

    packagePCDDataset(raw_test.tolist(), lab_test.tolist(), 'testing', output, scanlines)

    test_end = time()
    end = time()

    train_time = train_end - train_start
    val_time = val_end - val_start
    test_time = test_end - test_start
    total_time = end - start

    dataset.print_hdf5_info(output + '.hdf5')
    print('Time to build training set: {}s'.format(train_time))
    print('Time to build validation set: {}s'.format(val_time))
    print('Time to build testing set: {}s'.format(test_time))
    print('Total time to build set: {}s'.format(total_time))

    return


def createImgH5(raw, labeled, output, split):
    start = time()
    index1 = int(len(labeled) * split[0] / 100.0)
    index2 = int(len(labeled) * (split[0] + split[1]) / 100.0)

    raw, labeled = shuffle_in_unison(raw, labeled)

    lab_train, lab_val, lab_test = np.split(labeled, [index1, index2])
    raw_train, raw_val, raw_test = np.split(raw, [index1, index2])

    train_start = time()

    packageImgDataset(raw_train.tolist(), lab_train.tolist(), 'training', output)

    train_end = time()
    val_start = time()

    packageImgDataset(raw_val.tolist(), lab_val.tolist(), 'validation', output)

    val_end = time()
    test_start = time()

    packageImgDataset(raw_test.tolist(), lab_test.tolist(), 'testing', output)

    test_end = time()
    end = time()

    train_time = train_end - train_start
    val_time = val_end - val_start
    test_time = test_end - test_start
    total_time = end - start

    dataset.print_hdf5_info(output + '.hdf5')
    print('Time to build training set: {}s'.format(train_time))
    print('Time to build validation set: {}s'.format(val_time))
    print('Time to build testing set: {}s'.format(test_time))
    print('Total time to build set: {}s'.format(total_time))

    return
