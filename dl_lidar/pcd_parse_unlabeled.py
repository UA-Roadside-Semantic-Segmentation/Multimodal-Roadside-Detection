import glob
import numpy as np
import gzip

rows = 16
cols = 2048
channels = 4


def format_set(path, rows, cols, channels, name):
    data = np.empty(shape=(0, rows, cols, channels))
    header_rows = 11

    for pcd_file in sorted(glob.glob(path)):
        file_data = np.loadtxt(pcd_file, delimiter=' ', skiprows=header_rows)
        file_data_copy = file_data.copy()
        file_data_copy.resize((rows * cols, channels))
        formatted_data = np.reshape(file_data_copy, newshape=(cols, rows, channels))
        formatted_data = np.swapaxes(formatted_data, 0, 1)
        data = np.append(data, np.expand_dims(formatted_data, axis=0), axis=0)

    final_output = data[:, :, :, :]
    print('Final output shape: ' + str(final_output.shape))
    print('Saving: ' + name + '.npy.gz')

    f = gzip.GzipFile(name + '.npy.gz', 'w')
    np.save(file=f, arr=final_output, allow_pickle=False)
    f.close()


format_set(path='hi-res-labeled/*', rows=rows, cols=cols, channels=channels, name='training_data')
format_set(path='validation/raw_clouds/*', rows=rows, cols=cols, channels=channels, name='validation_data')
format_set(path='test/raw_clouds/*', rows=rows, cols=cols, channels=channels, name='test_data')

# to load
# f = gzip.GzipFile('organized_point_clouds.npy.gz', 'r')
# labeled_clouds = np.load(f)