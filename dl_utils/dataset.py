import cv2
import gzip
import numpy as np
import h5py
import pandas as pd


def img_to_ndarray(filepath):
    """
    Load an image into a numpy ndarray
    :param filepath: filepath of image to load. If image is not in working directory, absolute path must be used
    :return: ndarray containing bitmap of image. Will raise FileNotFoundError if file is not found.
    """
    print("Adding {} to ndarray".format(filepath))
    arr = cv2.imread(filepath)
    if arr is None:
        raise FileNotFoundError(f"Image at path {filepath} not found!")
    return arr


# extract the data from a pcd file
def pcd_to_ndarray(filepath, rows, cols, channels):
    """
    Load an organized pcd file into a numpy ndarray
    :param filepath: filepath of pcd to load
    :param rows: num of rows organized cloud has
    :param cols: num of cols organized cloud has
    :param channels: num of channels organized cloud has
    :return: ndarray containing organized point cloud projected into 2d plane
    """
    try:
        file_data = pd.read_csv(filepath, delimiter=' ', header=10, comment='#').to_numpy()
    except:
        raise RuntimeError('Error loading ' + filepath)
    file_data_copy = file_data.copy()
    file_data_copy.resize([rows * cols, channels], refcheck=False)
    formatted_data = np.reshape(file_data_copy, newshape=(cols, rows, channels))
    formatted_data = np.swapaxes(formatted_data, 0, 1)
    # print(type(formatted_data[0, 0, 0]))
    return formatted_data


def img_list_to_ndarray(filepaths):
    """
    Load a list of images into a numpy ndarray
    :param filepaths: filepaths of images to load
    :return: ndarray containing len(filepaths) bitmap of images
    """
    return np.array([img_to_ndarray(img_path) for img_path in filepaths])


def pcd_list_to_ndarray(filepaths, rows=16, cols=2048, channels=4):
    """
    Load a list of pcds into a numpy ndarray
    :param filepaths: filepaths of pcds to load
    :return: ndarray containing len(filepaths) organized pointclouds
    """
    return np.array([pcd_to_ndarray(pcd_path, rows=rows, cols=cols, channels=channels) for pcd_path in filepaths])

def txt_to_ndarray(filepath):
    with open(filepath, 'r') as f:
        print('Adding {} to ndarray'.format(filepath))
        return np.string_(f.read())


def txt_list_to_ndarray(filepaths):
    temp = np.empty((len(filepaths), 1))
    temp = [txt_to_ndarray(img_path) for img_path in filepaths]
    return temp


def save_ndarray_as_npy_gz(dataset, filename):
    """
    Save an ndarray as a compressed npy.gz archive
    :param dataset: name of ndarray to be saved
    :param filename: filename that npy.gz archive will be saved as.  !!!INCLUDE npy.gz IN FILENAME!!!
    :return: None
    """
    print(f'\nSaving {filename}')
    f = gzip.GzipFile(filename, 'w+')
    np.save(file=f, arr=dataset, allow_pickle=False)
    f.close()


def save_ndarray_as_npy(dataset, filename):
    print(f'\nSaving {filename}')
    np.save(file=filename, arr=dataset, allow_pickle=False)


def save_ndarray_as_npz(dataset, filename):
    print(f'\nSaving {filename}')
    np.savez_compressed(file=filename, arr=dataset, allow_pickle=False, )


def load_npy_gz(filename):
    """
    Load ndarray from compressed npy.gz archive
    :param filename: path of archived npy.gz archive
    :return: ndarray containing data from archive
    """
    print(f'\nLoading {filename}')
    f = gzip.GzipFile(filename, 'r')
    array = np.load(f)
    f.close()
    return array


def create_img_hdf5(datalist, filename, key, batchsize=1, compression=True):
    """
    Creates an hdf5 file from a list
    :param datalist: list; the list of file paths to be added to hdf5 file
    :param filename: string; output filename. This should be the relative path to the final output.
    :param key: string; the name for the key. (ex. 'Blackfly_Raw', 'Blackfly_Labeled', 'data')
    :param batchsize: int; how many items to add to the file from the list at a time. higher=more RAM but faster. does not
                        need to be divisible by list length
    :param compression: bool; if dataset should use gzip level compression. Default is true. False=More disk space, but faster
    """

    scan = img_to_ndarray(datalist.pop(0))

    hf = h5py.File(filename, 'a')

    if compression:
        dset = hf.create_dataset(key, maxshape=((None,) + scan.shape), shape=((1,) + scan.shape), compression='gzip', dtype=np.uint8)
    else:
        dset = hf.create_dataset(key, maxshape=((None,) + scan.shape), shape=((1,) + scan.shape), dtype=np.uint8)
    dset[:] = scan

    hf.close()

    if datalist:
        append_img_hdf5(datalist, filename, key, batchsize)

    return


def append_img_hdf5(datalist, filename, key, batchsize=1):
    """
    Adds new data to the end of a pre-existing dataset
    :param dataset: List of new file paths to be added
    :param filename: the hdf5 file to be added to
    :param key: the dataset within the hdf5 file to be added to
    :param batchsize: how many items to add to the file from the list at a time. higher=more RAM but faster. does not
                        need to be divisible by list length
    """

    datalist = [datalist[i * batchsize:(i + 1) * batchsize] for i in
                range((len(datalist) + batchsize - 1) // batchsize)]

    hf = h5py.File(filename, 'r+')
    dset = hf[key]

    for i in datalist:
        dset.resize(dset.shape[0] + len(i), axis=0)
        print('Resized to: {}'.format(dset.shape))
        arr = img_list_to_ndarray(i)
        dset[-len(i):] = arr

    print('Final dataset shape: {}'.format(dset.shape))
    hf.close()
    return


def add_img_dataset_hdf5(datalist, filename, key, batchsize=1, compression=True):
    """
    Adds an aditional dataset to an existing hdf5 file

    :param datalist: list; the list of file paths to be added to hdf5 file
    :param filename: string; output filename. This should be the relative path to the final output. *BE SURE TO INCLUDE '.HDF5'*
    :param key: string; the name for the key. (ex. 'Blackfly_Raw', 'Blackfly_Labeled', 'data') *NEEDS TO BE DIFFERENT FROM CURRENT KEYS*
    :param batchsize: int; how many items to add to the file from the list at a time. higher=more RAM but faster. does not
                        need to be divisible by list length
    :param compression: bool; if dataset should use gzip level compression. Default is true. False=More disk space, but faster
    """

    create_img_hdf5(datalist, filename, key, batchsize, compression)
    return


def create_txt_hdf5(datalist, filename, key, batchsize=1, compression=True):
    """

    :param datalist:
    :param filename:
    :param key:
    :param batchsize:
    :param compression:
    :return:
    """

    scan = txt_to_ndarray(datalist.pop(0))

    hf = h5py.File(filename, 'a')

    # I have no idea what 'S54' means. But it's what the dtype is set to when I try adding data immediately, instead of creating an empty dset
    if compression:
        dset = hf.create_dataset(key, maxshape=(None, ), shape=(1, ), compression='gzip', dtype='S54')
    else:
        dset = hf.create_dataset(key, maxshape=(None, ), shape=(1, ), dtype='S54')
    dset[:] = scan

    hf.close()

    if datalist:
        append_txt_hdf5(datalist, filename, key, batchsize)

    return


def append_txt_hdf5(datalist, filename, key, batchsize=1, compression=True):
    """

    :param datalist:
    :param filename:
    :param key:
    :param batchsize:
    :param compression:
    :return:
    """

    datalist = [datalist[i * batchsize:(i + 1) * batchsize] for i in
                range((len(datalist) + batchsize - 1) // batchsize)]

    hf = h5py.File(filename, 'r+')
    dset = hf[key]

    for i in datalist:
        dset.resize(dset.shape[0] + len(i), axis=0)
        print('Resized to: {}'.format(dset.shape))
        arr = txt_list_to_ndarray(i)
        dset[-len(i):] = arr

    print('Final dataset shape: {}'.format(dset.shape))
    hf.close()
    return


def add_txt_dataset_hdf5(datalist, filename, key, batchsize=1, compression=True):
    """

    :param datalist:
    :param filename:
    :param key:
    :param batchsize:
    :param compression:
    :return:
    """

    create_txt_hdf5(datalist, filename, key, batchsize, compression)
    return


def create_pcd_hdf5(datalist, filename, key, batchsize=1, compression=True, rows=16, columns=2048, channels=4):
    """
    Creates an hdf5 file from a list

    :param datalist: list; the list of file paths to be added to hdf5 file
    :param filename: string; output filename. This should be the relative path to the final output.
    :param key: string; the name for the key. (ex. 'Blackfly_Raw', 'Blackfly_Labeled', 'data')
    :param batchsize: int; how many items to add to the file from the list at a time. higher=more RAM but faster. does not
                        need to be divisible by list length
    :param compression: bool; if dataset should use gzip level compression. Default is true. False=More disk space, but faster
    :param rows: how many scan lines are in the PCD file
    :param columns: how many points are in each scan line. typically 2048 for all our lidars
    :param channels: how much data is returned per PCD point. raw(4): (x,y,z,intensity) labeled(5): (x,y,z,label,tag)
    """

    scan = pcd_to_ndarray(datalist.pop(0), rows, columns, channels)

    hf = h5py.File(filename, 'a')

    if channels == 4:
        if compression:
            dset = hf.create_dataset(key, maxshape=(None, rows, columns, channels), shape=(1, rows, columns, channels), compression='gzip')
        else:
            dset = hf.create_dataset(key, maxshape=(None, rows, columns, channels), shape=(1, rows, columns, channels))
        dset[:] = scan
    else:
        if compression:
            dset = hf.create_dataset(key, maxshape=(None, rows, columns), shape=(1, rows, columns), compression='gzip')
        else:
            dset = hf.create_dataset(key, maxshape=(None, rows, columns), shape=(1, rows, columns))
        dset[:] = scan[:, :, 3]
    hf.close()

    if datalist:
        append_pcd_hdf5(datalist, filename, key, batchsize, rows, columns, channels)

    return


def append_pcd_hdf5(datalist, filename, key, batchsize=1, rows=16, columns=2048, channels=4):
    """
    Adds new data to the end of a pre-existing dataset
    :param dataset: List of new file paths to be added
    :param filename: the hdf5 file to be added to
    :param key: the dataset within the hdf5 file to be added to
    :param batchsize: how many items to add to the file from the list at a time. higher=more RAM but faster. does not
                        need to be divisible by list length
    :param rows: how many scan lines are in the PCD file
    :param columns: how many points are in each scan line. typically 2048 for all our lidars
    :param channels: how much data is returned per PCD point. raw(4): (x,y,z,intensity) labeled(5): (x,y,z,label,tag)
    """

    datalist = [datalist[i * batchsize:(i + 1) * batchsize] for i in
                range((len(datalist) + batchsize - 1) // batchsize)]

    hf = h5py.File(filename, 'r+')
    dset = hf[key]

    for i in datalist:
        dset.resize(dset.shape[0] + len(i), axis=0)
        print('Resized to: {}'.format(dset.shape))
        arr = pcd_list_to_ndarray(i, rows, columns, channels)
        if channels == 4:
            dset[-len(i):] = arr[:, :, :]
        else:
            dset[-len(i):] = arr[:, :, :, 3]

    print('Final dataset shape: {}'.format(dset.shape))
    hf.close()
    return


def add_pcd_dataset_hdf5(datalist, filename, key, batchsize=1, compression=True, rows=16, columns=2048, channels=4):
    """
    Adds an aditional dataset to an existing hdf5 file

    :param datalist: list; the list of file paths to be added to hdf5 file
    :param filename: string; output filename. This should be the relative path to the final output. *BE SURE TO INCLUDE '.HDF5'*
    :param key: string; the name for the key. (ex. 'Blackfly_Raw', 'Blackfly_Labeled', 'data') *NEEDS TO BE DIFFERENT FROM CURRENT KEYS*
    :param batchsize: int; how many items to add to the file from the list at a time. higher=more RAM but faster. does not
                        need to be divisible by list length
    :param compression: bool; if dataset should use gzip level compression. Default is true. False=More disk space, but faster
    :param rows: how many scan lines are in the PCD file
    :param columns: how many points are in each scan line. typically 2048 for all our lidars
    :param channels: how much data is returned per PCD point. raw(4): (x,y,z,intensity) labeled(5): (x,y,z,label,tag)
    """

    create_pcd_hdf5(datalist, filename, key, batchsize, compression, rows, columns, channels)
    return


def print_hdf5_info(filename):
    """
    prints the keys and the shape of data within it
    :param filename: filepath to hdf5 file you want the info for
    """

    hf = h5py.File(filename, 'r')
    print('\nFilename: {}'.format(filename))
    for key in hf.keys():
        print('Key: {}\tSize: {}'.format(key, hf[key].shape))
    hf.close()
    return
