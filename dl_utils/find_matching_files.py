import numpy as np
import pandas as pd
import sys
import os


def find_lidar_timestamp(lidar_filename):
    i = np.where(lidar_ts[:, 0] == lidar_filename)
    return int(lidar_ts[i, 1])


def find_camera_timestamp(camera_filename):
    i = np.where(camera_ts[:, 0] == camera_filename)
    return int(camera_ts[i, 1])


def find_closest_camera_filename(timestamp):
    i = np.argmin(np.abs(camera_ts[:, 1] - timestamp))
    return camera_ts[i]


def find_closest_lidar_filename(timestamp, lidar_name):
    find_result = np.array([entry.find(lidar_name) for entry in lidar_ts[:, 0]])
    mask = np.where(find_result == 0)
    lidar_ts_masked = lidar_ts[mask]

    if lidar_ts_masked.size == 0:
        return 'no ' + lidar_name + ' scan found', timestamp

    i = np.argmin(np.abs(lidar_ts_masked[:, 1] - timestamp))
    return lidar_ts_masked[i]


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print('Usage: python {} timestamps_folder'.format(sys.argv[0]))
        sys.exit(1)

    lidar_ts = pd.read_csv(os.path.join(sys.argv[1], 'lidar_timestamps.txt'), delimiter=',', header=None).to_numpy()
    camera_ts = pd.read_csv(os.path.join(sys.argv[1], 'camera_timestamps.txt'), delimiter=',', header=None).to_numpy()

    lidar_names = ['os1', 'os2', 'puck', 'hi-res', '32mr']

    while True:
        filename = input('Enter a .jpg or .pcd filename: ')
        filename = os.path.basename(filename.strip(' \'\"'))

        if os.path.splitext(filename)[1] == '.pcd':
            timestamp = find_lidar_timestamp(filename)
            lidar_name = os.path.splitext(filename)[0].split('_')[0]
            print('Matching files:')

            # print blackfly filename (and ms difference)
            camera_filename, camera_timestamp = find_closest_camera_filename(timestamp)
            difference = camera_timestamp - timestamp
            print('\t{} ({:+d}ms)'.format(camera_filename, difference))

            # print other lidar sensor filenames (and ms differences)
            for lidar in lidar_names:
                if lidar == lidar_name:
                    continue
                lidar_filename, lidar_timestamp = find_closest_lidar_filename(timestamp, lidar)
                difference = lidar_timestamp - timestamp
                print('\t{} ({:+d}ms)'.format(lidar_filename, difference))

        elif os.path.splitext(filename)[1] == '.jpg':
            timestamp = find_camera_timestamp(filename)
            print('Matching files:')

            # print lidar filenames (and ms differences)
            for lidar in lidar_names:
                lidar_filename, lidar_timestamp = find_closest_lidar_filename(timestamp, lidar)
                difference = lidar_timestamp - timestamp
                print('\t{} ({:+d}ms)'.format(lidar_filename, difference))

        else:
            print('Not a .jpg or .pcd file!')
