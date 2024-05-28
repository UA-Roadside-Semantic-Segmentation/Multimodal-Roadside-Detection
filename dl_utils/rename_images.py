import os
import sys
import glob


if __name__ == '__main__':
    if len(sys.argv) != 6:
        print('Usage: python {} sensor_folder sensor_name site_number truck_or_train raw_or_labeled'.format(sys.argv[0]))
        sys.exit(1)

    sensor_folder = sys.argv[1]
    sensor_name = sys.argv[2]
    site_number = int(sys.argv[3])
    truck_or_train = sys.argv[4]
    raw_or_labeled = sys.argv[5]

    dest_folder = os.path.join(sensor_folder, 'new')
    extension = '.jpg' if sensor_name == 'blackfly' else '.pcd'

    # CHECK REVERSED T/F!
    images = sorted(glob.glob(os.path.join(sensor_folder, '*'+extension)), reverse=False)

    sys.stdout.write('Renaming {} files from {} '.format(len(images), sensor_folder))
    sys.stdout.flush()

    os.makedirs(dest_folder, exist_ok=True)

    counter = 0
    for image in images:
        new_name = '{}_{}_{:0>3}_{:0>5}_{}{}'.format(sensor_name, truck_or_train, site_number, counter, raw_or_labeled, extension)
        # new_name = '{}_{:0>3}_{:0>5}{}'.format(sensor_name, site_number, counter, extension)
        new_path = os.path.join(dest_folder, new_name)
        os.renames(image, new_path)

        sys.stdout.write('.')
        sys.stdout.flush()
        # print('{} => {}'.format(image, new_path))
        counter += 1

print()
