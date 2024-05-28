import os
import sys
import glob
import shutil
import errno
# import linecache
# from itertools import islice

"""
def bad_width(file):
    # width_line = linecache.getline(file, 6)
    # width = int(width_line.split()[1])
    # if width < 2000:
    #     return True
    # return False

    # with open(file) as lines:
    #     for width_line in islice(lines, 6, 7):
    #         width = int(width_line.split()[1])
    #         if width < 2000:
    #             return True
    #         return False

    # this one is fastest
    for i, line in enumerate(open(file)):
        if (i == 7 and line.strip().split()[0] == 'POINTS') or i == 9:
            points = int(line.strip().split()[1])
            if points < 2000 * 32: # specific to 32mr!
                return True
            return False
"""


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python {} sensor_folder sensor_name dest_folder'.format(sys.argv[0]))
        sys.exit(1)

    train_folders = sorted(glob.glob(os.path.join(sys.argv[1], '*', '')))
    sensor_name = sys.argv[2]
    dest_folder = sys.argv[3]

    for folder in train_folders:

        train = int(os.path.basename(os.path.normpath(folder))[-3:])
        # CHECK REVERSED T/F!
        images = sorted(glob.glob(os.path.join(folder, '*.jpg' if sensor_name == 'blackfly' else '*.pcd')), reverse=True)

        sys.stdout.write('Renaming {} files from {} '.format(len(images), folder))
        sys.stdout.flush()

        counter = 0
        for image in images:

            # if sensor_name != 'blackfly' and bad_width(image):
            #     sys.stdout.write('x')
            #     sys.stdout.flush()
            #     continue

            extension = os.path.splitext(image)[1]
            new_name = '{}_{:0>3}_{:0>5}{}'.format(sensor_name, train, counter, extension)
            new_folder = os.path.join(dest_folder, 't{:0>3}'.format(train), 'raw')
            new_path = os.path.join(new_folder, new_name)

            try:
                os.makedirs(new_folder)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            shutil.copy2(image, new_path)
            sys.stdout.write('.')
            sys.stdout.flush()
            # print('{} => {}'.format(image, new_path))
            counter += 1

        print()
