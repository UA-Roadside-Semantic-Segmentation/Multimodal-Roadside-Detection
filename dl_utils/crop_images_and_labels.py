# crop_images_and_labels.py
# Authors: Andrew Sucato and Matt Mason
#
# This tool takes a folder containing images (>600 vertical pixels) and labels sizes the images down
# to 600 vertical pixels.
#
# Specifically, this tool finds the average vertical centerline of all the labels
# for an image, and crops the image to 300 pixels above and below that average centerline.  If the centerline
# found is within 300 pixels of the top or bottom of the image, the centerline is clamped to 300 pixels from the
# top or bottom.  The box labels are then shifted to be in the correct position for the cropped image using the
# formula: y_shift = -(centerline - 300), where 300 is the new centerline of the cropped image.
#
# The tool only adjusts image/label pairs, and ignores images that do not have a corresponding label.
#
# The tool will output the cropped images and labels in a new folder per folder passed as an argument, with
# '_cropped' appended to the folder name.  For ex. if you run 'python3 crop_images_and_labels.py ~/t001 ~/t002',
# it will put the adjusted images and labels in ~/t001_cropped and ~/t002_cropped.


import dataset
import numpy as np
import contextlib
import glob
import sys
import os
import cv2

if len(sys.argv) < 2:
    print('Usage: python3 {} train_folder/ truck_folder/ ...'.format(sys.argv[0]))
    print('   Ex: python3 {} t000/ t001/ t002/'.format(sys.argv[0]))
    sys.exit(1)

folders = sys.argv[1:]

for folder in folders:
    label_files = glob.glob(os.path.join(folder, 'labeled', '*.txt'))
    raw_files = list()
    for t in label_files:
        check = t.split('/')[-1].split('.')[0].split('_')
        del (check[-1])
        check = '_'.join(check)
        raw_files += glob.glob(os.path.join(folder, 'raw', '{}.jpg'.format(check)))
        raw_files += glob.glob(os.path.join(folder, 'raw', '{}.png'.format(check)))

    # find average centerline of labels
    centerline = 0
    total_boxes = 0
    for label_file in label_files:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            label = dataset.txt_to_ndarray(label_file)
        label = label.decode('UTF-8')

        boxes = label.split('\n')
        del(boxes[-1])
        boxes = np.array([np.fromstring(b, dtype=int, sep=',') for b in boxes])

        for box in boxes:
            y_midpoint = (2*box[1] + box[3]) / 2
            centerline += y_midpoint
            total_boxes += 1

    centerline = int(centerline / total_boxes)
    print('Average centerline for {} is {}'.format(folder, centerline))

    # crop images to 600x2048 and shift labels accordingly
    for image_file, label_file in zip(raw_files, label_files):
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            image = dataset.img_to_ndarray(image_file)
            label = dataset.txt_to_ndarray(label_file)

        print('image shape: {}'.format(image.shape))
        height = image.shape[0]
        if height < 600:
            raise ValueError('Image height is {}, which is less than 600'.format(height))

        # crop image to centerline +/- 300, but handle correctly if (centerline - 300) < 0 or (centerline + 300) > image height
        y_shift = 0
        if centerline <= 300:
            centerline = 300
        elif (centerline + 300) > height:
            centerline = height - 300

        start = centerline - 300
        image = image[start:start+600]
        y_shift = -(centerline - 300)

        print('\nBoxes: {}'.format(boxes))

        # loop through boxes
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            label = dataset.txt_to_ndarray(label_file)
        label = label.decode('UTF-8')

        boxes = label.split('\n')
        del (boxes[-1])
        boxes = np.array([np.fromstring(b, dtype=int, sep=',') for b in boxes])
        # shift box y coordinates based on image crop
        if boxes.size > 0:
            new_boxes = boxes + [0, y_shift, 0, 0]

        cropped_folder = os.path.dirname(folder) + '_cropped/'
        os.makedirs(cropped_folder + 'labels', exist_ok=True)
        os.makedirs(cropped_folder + 'raw', exist_ok=True)
        # save new copy of image in folder+_cropped/raw/
        if boxes.size > 0:
            np.savetxt(cropped_folder + 'labels/' + os.path.basename(label_file), new_boxes, delimiter=',', fmt='%d')
        else:
            open(cropped_folder + 'labels/' + os.path.basename(label_file), 'w').close()
        # save new copy of label in folder+_cropped/labels/
        cv2.imwrite(cropped_folder + 'raw/' + os.path.basename(image_file), image)
        print(image.shape)
