import cv2
import h5py
import sys
# import tensorflow as tf
import numpy as np

if len(sys.argv) != 2:
    print('Usage: python3 {} dataset'.format(sys.argv[0]))
    sys.exit()

hf = h5py.File(sys.argv[1], 'r')
raw = hf['img']
inferred = hf['inferred']

def load_img(idx):
    raw_image = (raw[idx]+1)
    mask = (inferred[idx, :, :, 0]*255).astype(np.uint8)
    return raw_image, mask

inc = 1
imgIdx = 0

raw_image, mask = load_img(imgIdx)

cv2.namedWindow('Raw Image')
cv2.namedWindow('Inference Mask')

cv2.imshow('Raw Image', raw_image)
cv2.imshow('Inference Mask', mask)

while 1:
    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break
    elif key == ord('\r'):
        print(imgIdx)
        imgIdx += 1
        raw_image, mask = load_img(imgIdx)
        cv2.imshow('Raw Image', raw_image)
        cv2.imshow('Inference Mask', mask)

cv2.destroyAllWindows()