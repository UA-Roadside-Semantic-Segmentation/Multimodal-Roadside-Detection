import cv2
import numpy as np
import dl_utils.dataset as utils
import h5py
import sys
from time import time


def apply_rand_transform(img, mask):
    img = img + np.random.normal(0.0, 3, img.shape)

    right = int(np.random.uniform(0, 250))
    left = int(np.random.uniform(0, 250))
    top = int(np.random.uniform(0, 100))
    bottom = int(np.random.uniform(0, 100))

    img_crop = img[top:(img.shape[0]-bottom), left:(img.shape[1]-right)]
    mask_crop = mask[top:(img.shape[0]-bottom), left:(img.shape[1]-right)]

    img = cv2.resize(img_crop, (img.shape[1], img.shape[0]))
    mask = cv2.resize(mask_crop, (img.shape[1], img.shape[0]))

    if np.random.choice([True, False]):
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    return img, mask



def image_data_generator(data, ttv, batchSize, output_res, augment):
    ttv_compare = ['testing', 'training', 'validation']
    if ttv not in ttv_compare:
        print('Error in image_data_gen: ttv must be exactly "testing", "training", or "validation"')
        sys.exit(1)

    hf = h5py.File(data, 'r')
    f_data = hf[ttv + '_data']
    f_labels = hf[ttv + '_labeled']

    print(f_data.shape)
    print(f_labels.shape)

    while True:
        indices = np.random.choice(f_data.shape[0], size=batchSize, replace=False)
        indices.sort()  # ensures the random indices are in ascending order. limitation of h5

        x = np.empty((batchSize,) + output_res + (3,))
        y = np.empty((batchSize,) + (output_res[0], output_res[1], 2))

        in_res = [len(f_data[0]), len(f_data[0][0])]

        data = np.empty([batchSize, in_res[0], in_res[1], 3], dtype='uint8')
        labels = np.empty([batchSize], dtype='S54')

        for c, i in enumerate(indices):
            data[c] = f_data[i]
            labels[c] = f_labels[i]

        for idx, i in enumerate(labels):
            img = data[idx]
            i = i.decode('UTF-8')
            boxes = i.split('\n')
            del(boxes[-1])
            boxes = np.array([np.fromstring(b, dtype=int, sep=',') for b in boxes])
            numBoxes = boxes.shape[0]
            img_mask = np.zeros((in_res[0], in_res[1]))

            for i in range(0, numBoxes):
                cv2.rectangle(img_mask, tuple(boxes[i, [0, 1]]), tuple(boxes[i, [0, 1]] + boxes[i, [2, 3]]), 255, -1)

            if augment:
                transformed, transformed_mask = apply_rand_transform(img, img_mask)
            else:
                transformed, transformed_mask = img, img_mask

            #print (img.shape)

            # cv2.imshow('win2', transformed)
            # cv2.imshow('win', transformed_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            resized = (cv2.resize(transformed, (output_res[1], output_res[0])) / 127.5) - 1.0
            resized_mask = cv2.resize(transformed_mask, (output_res[1], output_res[0])) / 255

            reshaped_mask = np.reshape(resized_mask, (output_res[0], output_res[1]))
            inverseMask = 1 - reshaped_mask

            train_classes = np.array([reshaped_mask, inverseMask])
            train_classes = np.moveaxis(train_classes, 0, 2)

            x[idx] = resized
            y[idx] = train_classes

        yield x, y


"""Used to benchmark changes to the data generator"""

