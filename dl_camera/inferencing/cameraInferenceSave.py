import cv2
import h5py
import sys
import numpy as np

if len(sys.argv) != 4:
    print("Usage: python3 {} dataset save_location index".format(sys.argv[0]))
    sys.exit()

index = int(sys.argv[3])
filename = sys.argv[2]

hf = h5py.File(sys.argv[1], 'r')
raw = hf['img']
inferred = hf['inferred']

img = (raw[index] + 1) * 255
mask = inferred[index, :, :, 0] * 255

cv2.imwrite(filename + '_raw.png', img)
cv2.imwrite(filename + '_mask.png', mask)
