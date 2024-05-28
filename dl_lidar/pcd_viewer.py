import sys
import os
import glob
import numpy as np
from dl_lidar.dataset_viewer import DatasetViewer
from create_datasets import parse_file


if not (len(sys.argv) == 3 or len(sys.argv) == 4):
    print('Usage: python3 {} scanlines unlabeled_folder [labeled_folder]'.format(sys.argv[0]))
    sys.exit(1)

rows = int(sys.argv[1])
unlabeled_folder = sys.argv[2]
labeled_folder = None
no_labels = False
if len(sys.argv) == 4:
    labeled_folder = sys.argv[3]
else:
    no_labels = True

cols = 2048
unlabeled_channels = 4
labeled_channels = 5

folder = unlabeled_folder if no_labels else labeled_folder
dataset = sorted(glob.glob(os.path.join(folder, '*.pcd')))

unlabeled_data = np.empty(shape=(len(dataset), rows, cols, unlabeled_channels))
labeled_data = np.empty(shape=(len(dataset), rows, cols, labeled_channels))

sys.stdout.write('Parsing {} pcd files'.format(len(dataset)))
for i, file in enumerate(dataset):
    sys.stdout.write('.')
    sys.stdout.flush()
    if no_labels:
        unlabeled_data[i] = parse_file(file, rows, cols, unlabeled_channels)

    else:
        unlabeled_file = os.path.join(unlabeled_folder, os.path.basename(file))

        unlabeled_data[i] = parse_file(unlabeled_file, rows, cols, unlabeled_channels)
        labeled_data[i] = parse_file(file, rows, cols, labeled_channels)
print()

data = unlabeled_data[:, :, :, :]
labels = np.zeros(shape=unlabeled_data[:, :, :, 0].shape) if no_labels else labeled_data[:, :, :, 3]

dv = DatasetViewer(data, labels, folder, rows)
dv.show()
