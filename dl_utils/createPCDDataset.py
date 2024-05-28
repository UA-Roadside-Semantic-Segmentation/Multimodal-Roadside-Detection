import PCDH5Creator
import glob
import sys

dset_percentages = [80, 10, 10]

if len(sys.argv) != 5:
    print('Usage: python3 {} raw_directory_path labels_directory_path output_directory_path.hdf5 scanlines'.format(sys.argv[0]))
    sys.exit(1)

if sum(dset_percentages) != 100:
    print('train_validate_test_percentages must add up to 100')
    sys.exit(1)

scanlines = int(sys.argv[4])
output = sys.argv[3]

label_files = glob.glob(sys.argv[2] + '*.pcd')
raw_files = list()

for t in label_files:
    check = t.split('/')[-1]
    raw_files += glob.glob(sys.argv[1] + check)

PCDH5Creator.createH5(raw_files, label_files, output, scanlines, dset_percentages)